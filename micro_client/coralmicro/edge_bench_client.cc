#include "edge_bench_client.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "coralmicro/libs/tensorflow/utils.h"
#include "coralmicro/libs/tpu/edgetpu_manager.h"
#include "coralmicro/libs/tpu/edgetpu_op.h"
// #include "heap_manager.h"
#include <stdio.h>
#include <stdarg.h>
#include <set>

static const char *TAG = "EdgeBenchClient";

static constexpr int kMaxModelSize = 4488760;  // 4.5 MB
static constexpr int kArenaSize = 3200 * 1024; // 3.2 MB

static EdgeBenchClient *s_edgeBenchClient = nullptr;
static std::vector<uint8_t> s_model_buffer = std::vector<uint8_t>(kMaxModelSize);
static int s_model_size = 0;
static std::set<int> s_received_model_chunk_ids = std::set<int>();
static int s_input_size = 0;
static std::set<int> s_received_input_chunk_ids = std::set<int>();
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena_, kArenaSize);

void ESP_LOGI(const char *tag, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[%s] ", tag);
    vfprintf(stdout, format, args);
    printf("\r\n");
    va_end(args);
}

void ESP_LOGE(const char *tag, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[%s] ERROR: ", tag);
    vfprintf(stderr, format, args);
    printf("\r\n");
    va_end(args);
}

void messageArrivedHandler(MQTT::MessageData &md)
{
    if (s_edgeBenchClient == nullptr)
    {
        ESP_LOGE(TAG, "g_edgeBenchClient is null");
        return;
    }

    MQTT::Message &message = md.message;
    std::string topic(md.topicName.lenstring.data, md.topicName.lenstring.len);
    auto payload = static_cast<uint8_t *>(message.payload);
    std::vector<uint8_t> payloadVec(payload, payload + message.payloadlen);
    s_edgeBenchClient->handleMessage(topic, std::move(payloadVec));
}

EdgeBenchClient::EdgeBenchClient(const std::string &device_id,
                                 const std::string &broker_host,
                                 int broker_port)
    : device_id_(device_id),
      broker_host_(broker_host),
      broker_port_(broker_port),
      topic_(device_id)
{
    s_edgeBenchClient = this;
    int sockfd = ipstack_.connect(broker_host_.c_str(), broker_port_);
    if (sockfd < 0)
    {
        ESP_LOGE(TAG, "SocketClient failed to connect to %s:%d",
                 broker_host_.c_str(), broker_port_);
        return;
    }
    ESP_LOGI(TAG, "SocketClient connected to %s:%d", broker_host_.c_str(), broker_port_);
    mqttClient_ = new MQTT::Client<CoralIPStack, CoralTimer, kMqttMaxPayloadSize, 6>(ipstack_);
}

void EdgeBenchClient::connect()
{
    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 4;
    data.clientID.cstring = (char *)device_id_.c_str();
    auto rc = mqttClient_->connect(data);
    if (rc != 0)
    {
        ESP_LOGE(TAG, "MQTT connect failed with rc: %d", rc);
        return;
    }
    ESP_LOGI(TAG, "MQTT connected to %s:%d", broker_host_.c_str(), broker_port_);
}

void EdgeBenchClient::disconnect()
{
    mqttClient_->disconnect();
    ipstack_.disconnect();
    ESP_LOGI(TAG, "MQTT disconnected");
}

int EdgeBenchClient::publishMQTTMessage(const std::string &topic,
                                        const uint8_t *payload,
                                        size_t payload_len,
                                        MQTT::QoS qos)
{
    MQTT::Message message;
    message.qos = qos;
    message.retained = false;
    message.dup = false;
    message.payload = (void *)payload;
    message.payloadlen = payload_len;

    int rc = mqttClient_->publish(topic.c_str(), message);
    return rc;
}

void EdgeBenchClient::sendStatus(ClientStatus status)
{
    uint8_t b = static_cast<uint8_t>(status);
    int rc = publishMQTTMessage(topic_.STATUS(), &b, sizeof(b));
    if (rc != 0)
    {
        ESP_LOGE(TAG, "Failed to send status message: %d", rc);
        return;
    }
    ESP_LOGI(TAG, "Status sent: %d", b);
}

void EdgeBenchClient::sendResult(int elapsed_time_ms)
{
    uint8_t buf[4] = {
        uint8_t(elapsed_time_ms >> 24),
        uint8_t(elapsed_time_ms >> 16),
        uint8_t(elapsed_time_ms >> 8),
        uint8_t(elapsed_time_ms)};
    int rc = publishMQTTMessage(topic_.RESULT_LATENCY(), buf, sizeof(buf));
    if (rc != 0)
    {
        ESP_LOGE(TAG, "Failed to send result message: %d", rc);
        return;
    }
    ESP_LOGI(TAG, "Result sent: %d ms", elapsed_time_ms);
}

void EdgeBenchClient::startLatencyTest()
{
    ESP_LOGI(TAG, "Running %d iterations...", iterations_);
    auto t0 = xTaskGetTickCount() * portTICK_PERIOD_MS;
    for (int i = 0; i < iterations_; ++i)
    {
        interpreter_->Invoke();
    }
    auto t1 = xTaskGetTickCount() * portTICK_PERIOD_MS;
    sendStatus(ClientStatus::DONE);
    int ms = t1 - t0;
    ESP_LOGI(TAG, "Run completed: %d ms", ms);
    sendResult(ms);
}

void EdgeBenchClient::startAccuracyTest()
{
    interpreter_->Invoke();
    size_t out_bytes = interpreter_->output_tensor(0)->bytes;

    // Send the output result in chunks
    // Chunk format: total_chunks (4 bytes), chunk_id (4 bytes), offset (4 bytes), chunk_data (remaining bytes)
    static constexpr int kChunkSize = 12800-64;
    int total_chunks = (out_bytes + kChunkSize - 1) / kChunkSize;
    ESP_LOGI(TAG, "Output size: %d bytes, total chunks: %d", out_bytes, total_chunks);
    for (int chunk_id = 0; chunk_id < total_chunks; ++chunk_id)
    {
        int offset = chunk_id * kChunkSize;
        int chunk_size = kChunkSize <= out_bytes - offset ? kChunkSize : out_bytes - offset;
        std::vector<uint8_t> chunk_data(chunk_size + 12);
        chunk_data[0] = (total_chunks >> 24) & 0xFF;
        chunk_data[1] = (total_chunks >> 16) & 0xFF;
        chunk_data[2] = (total_chunks >> 8) & 0xFF;
        chunk_data[3] = total_chunks & 0xFF;
        chunk_data[4] = (chunk_id >> 24) & 0xFF;
        chunk_data[5] = (chunk_id >> 16) & 0xFF;
        chunk_data[6] = (chunk_id >> 8) & 0xFF;
        chunk_data[7] = chunk_id & 0xFF;
        chunk_data[8] = (offset >> 24) & 0xFF;
        chunk_data[9] = (offset >> 16) & 0xFF;
        chunk_data[10] = (offset >> 8) & 0xFF;
        chunk_data[11] = offset & 0xFF;

        memcpy(chunk_data.data() + 12, output_tensor_ + offset, chunk_size);
        
        int rc = publishMQTTMessage(topic_.RESULT_ACCURACY(), chunk_data.data(), chunk_data.size());
        if (rc != 0)
        {
            ESP_LOGE(TAG, "Failed to send accuracy result message: %d", rc);
            return;
        }
    }
    ESP_LOGI(TAG, "Sent accuracy result in %d chunks", total_chunks);
}

void EdgeBenchClient::subscribeToTopics()
{
    mqttClient_->setDefaultMessageHandler(messageArrivedHandler);
    int rc[6] = {
        mqttClient_->subscribe(topic_.CONFIG_MODE().c_str(), MQTT::QOS1, messageArrivedHandler),
        mqttClient_->subscribe(topic_.CONFIG_ITERATIONS().c_str(), MQTT::QOS1, messageArrivedHandler),
        mqttClient_->subscribe(topic_.MODEL().c_str(), MQTT::QOS1, messageArrivedHandler),
        mqttClient_->subscribe(topic_.INPUT_LATENCY().c_str(), MQTT::QOS1, messageArrivedHandler),
        mqttClient_->subscribe(topic_.INPUT_ACCURACY().c_str(), MQTT::QOS1, messageArrivedHandler),
        mqttClient_->subscribe(topic_.CMD().c_str(), MQTT::QOS1, messageArrivedHandler)};

    for (int i = 0; i < 6; ++i)
    {
        if (rc[i] != 0)
        {
            ESP_LOGE(TAG, "Failed to subscribe to topic %d with rc: %d", i, rc[i]);
            return;
        }
    }

    ESP_LOGI(TAG, "Subscribed to topics");
}

void EdgeBenchClient::handleMessage(const std::string &topic, const std::vector<uint8_t> &payload)
{
    ESP_LOGI(TAG, "Got message on '%s' (%d bytes)",
             topic.c_str(),
             (int)payload.size());
    if (topic == topic_.CONFIG_MODE())
    {
        if (payload.size() != 1)
        {
            ESP_LOGE(TAG, "Invalid mode message");
            quit_ = true;
            return;
        }
        mode_ = static_cast<TestMode>(payload[0]);
        ESP_LOGI(TAG, "Mode set: %d", int(mode_));
    }
    else if (topic == topic_.CONFIG_ITERATIONS())
    {
        if (payload.size() != 4)
        {
            ESP_LOGE(TAG, "Invalid iterations message");
            quit_ = true;
            return;
        }
        // Assume big-endian 4-byte integer
        iterations_ =
            (static_cast<int>(payload[0]) << 24) |
            (static_cast<int>(payload[1]) << 16) |
            (static_cast<int>(payload[2]) << 8) |
            static_cast<int>(payload[3]);
        ESP_LOGI(TAG, "Iterations set: %d", iterations_);
    }
    else if (topic == topic_.MODEL())
    {
        ESP_LOGE(TAG, "Model chunk data received");
        // Chunk format: total_chunks (4 bytes), chunk_id (4 bytes), offset (4 bytes), chunk_data (remaining bytes)
        if (payload.size() < 12)
        {
            ESP_LOGE(TAG, "Invalid model chunk size");
            quit_ = true;
            return;
        }
        int total_chunks = (static_cast<int>(payload[0]) << 24) |
                           (static_cast<int>(payload[1]) << 16) |
                           (static_cast<int>(payload[2]) << 8) |
                           static_cast<int>(payload[3]);
        int chunk_id = (static_cast<int>(payload[4]) << 24) |
                       (static_cast<int>(payload[5]) << 16) |
                       (static_cast<int>(payload[6]) << 8) |
                       static_cast<int>(payload[7]);
        int offset = (static_cast<int>(payload[8]) << 24) |
                     (static_cast<int>(payload[9]) << 16) |
                     (static_cast<int>(payload[10]) << 8) |
                     static_cast<int>(payload[11]);
        ESP_LOGI(TAG, "Write model chunk %d/%d at offset %d", chunk_id, total_chunks, offset);
        memcpy(s_model_buffer.data() + offset, payload.data() + 12, payload.size() - 12);
        s_model_size += payload.size() - 12;
        s_received_model_chunk_ids.insert(chunk_id);

        if ((int)s_received_model_chunk_ids.size() == total_chunks)
        {
            ESP_LOGI(TAG, "All model chunks received (%d bytes)", s_model_size);

            const tflite::Model *model = tflite::GetModel(s_model_buffer.data());
            if (model->version() != TFLITE_SCHEMA_VERSION)
            {
                ESP_LOGE(TAG, "Model schema v%lu not supported", (unsigned long)model->version());
                quit_ = true;
                return;
            }

            delete interpreter_;

            if (tensor_arena_ == nullptr)
            {
                ESP_LOGE(TAG, "Tensor arena not allocated");
                quit_ = true;
                return;
            }

            if (micro_op_resolver_ == nullptr)
            {
                ESP_LOGE(TAG, "Micro op resolver not allocated");
                quit_ = true;
                return;
            }

            interpreter_ = new tflite::MicroInterpreter(model,
                                                        *micro_op_resolver_,
                                                        tensor_arena_,
                                                        kArenaSize,
                                                        &error_reporter_);
            TfLiteStatus allocate_status = interpreter_->AllocateTensors();
            if (allocate_status != kTfLiteOk)
            {
                ESP_LOGE(TAG, "AllocateTensors() failed");
                quit_ = true;
                return;
            }
            input_tensor_ = interpreter_->typed_input_tensor<int8_t>(0);
            output_tensor_ = interpreter_->typed_output_tensor<int8_t>(0);
            ESP_LOGI(TAG, "Model loaded; tensors allocated");
            ESP_LOGI(TAG, "Copy input data to tensor");
            model_input_size_ = interpreter_->input_tensor(0)->bytes;
        }
        else
        {
            sendStatus(ClientStatus::READY_FOR_CHUNK);
        }
    }
    else if (topic == topic_.INPUT_LATENCY() || topic == topic_.INPUT_ACCURACY())
    {
        ESP_LOGI(TAG, "Input chunk data received");
        // Chunk format: total_chunks (4 bytes), chunk_id (4 bytes), offset (4 bytes), chunk_data (remaining bytes)
        if (payload.size() < 12)
        {
            ESP_LOGE(TAG, "Invalid input chunk size");
            quit_ = true;
            return;
        }
        int total_chunks = (static_cast<int>(payload[0]) << 24) |
                           (static_cast<int>(payload[1]) << 16) |
                           (static_cast<int>(payload[2]) << 8) |
                           static_cast<int>(payload[3]);
        int chunk_id = (static_cast<int>(payload[4]) << 24) |
                       (static_cast<int>(payload[5]) << 16) |
                       (static_cast<int>(payload[6]) << 8) |
                       static_cast<int>(payload[7]);
        int offset = (static_cast<int>(payload[8]) << 24) |
                     (static_cast<int>(payload[9]) << 16) |
                     (static_cast<int>(payload[10]) << 8) |
                     static_cast<int>(payload[11]);
        ESP_LOGI(TAG, "Write input chunk %d/%d at offset %d", chunk_id, total_chunks, offset);

        if (input_tensor_ == nullptr)
        {
            ESP_LOGE(TAG, "Input tensor is null");
            quit_ = true;
            return;
        }

        memcpy(input_tensor_ + offset, payload.data() + 12, payload.size() - 12);
        s_input_size += payload.size() - 12;
        s_received_input_chunk_ids.insert(chunk_id);
        if ((int)s_received_input_chunk_ids.size() == total_chunks)
        {
            ESP_LOGI(TAG, "All input chunks received (%d bytes)", s_input_size);
            if (s_input_size != (int)model_input_size_) {
                ESP_LOGE(TAG, "Invalid input data size, expected %zu, got %zu",
                        model_input_size_,
                        s_input_size);
                quit_ = true;
                return;
            }
            s_input_size = 0;
            s_received_input_chunk_ids.clear();
            if (topic == topic_.INPUT_LATENCY())
            {
                latency_input_ready_ = true;
            }
            else if (topic == topic_.INPUT_ACCURACY())
            {
                startAccuracyTest();
            }
        }
        else
        {
            sendStatus(ClientStatus::READY_FOR_CHUNK);
        }
    }
    else if (topic == topic_.CMD())
    {
        if (payload.size() != 1)
        {
            ESP_LOGE(TAG, "Invalid command message");
            quit_ = true;
            return;
        }
        Command cmd = static_cast<Command>(payload[0]);
        if (cmd == Command::START_LATENCY_TEST)
        {
            startLatencyTest();
        }
        else if (cmd == Command::STOP)
        {
            quit_ = true;
            return;
        }
        else if (cmd == Command::RESET)
        {
            sent_ready_for_model_ = false;
            sent_ready_for_input_ = false;
            sent_ready_for_task_ = false;
            iterations_ = 0;
            latency_input_ready_ = false;
            mode_ = TestMode::NONE;
            delete interpreter_;
            interpreter_ = nullptr;
            s_model_buffer.clear();
            s_model_size = 0;
            s_received_model_chunk_ids.clear();
            s_input_size = 0;
            s_received_input_chunk_ids.clear();
            ESP_LOGI(TAG, "State reset");
            return;
        }
    }

    bool latency_config_ready =
        mode_ == TestMode::LATENCY && iterations_ > 0;

    bool accuracy_config_ready =
        mode_ == TestMode::ACCURACY;

    bool config_ready = latency_config_ready || accuracy_config_ready;
    bool interpreter_ready = interpreter_ != nullptr;
    bool input_ready = latency_input_ready_ || mode_ == TestMode::ACCURACY;

    if (!sent_ready_for_model_ && config_ready)
    {
        sent_ready_for_model_ = true;
        ESP_LOGI(TAG, "All config received, requesting model");
        sendStatus(ClientStatus::READY_FOR_MODEL);
    }

    if (!sent_ready_for_input_ && config_ready && interpreter_ready && mode_ == TestMode::LATENCY)
    {
        sent_ready_for_input_ = true;
        ESP_LOGI(TAG, "Interpreter ready, waiting for input");
        sendStatus(ClientStatus::READY_FOR_INPUT);
    }

    if (!sent_ready_for_task_ && config_ready && interpreter_ready && input_ready)
    {
        sent_ready_for_task_ = true;
        ESP_LOGI(TAG, "Ready for task, waiting for input or command");
        sendStatus(ClientStatus::READY_FOR_TASK);
    }
}

void EdgeBenchClient::run()
{
    ESP_LOGI(TAG, "Starting EdgeBenchClient for device %s", device_id_.c_str());
    auto tpu_context = coralmicro::EdgeTpuManager::GetSingleton()->OpenDevice();
    if (!tpu_context)
    {
        ESP_LOGE(TAG, "Failed to open Edge TPU device");
        return;
    }

    micro_op_resolver_ = new tflite::MicroMutableOpResolver<1>();
    micro_op_resolver_->AddCustom(coralmicro::kCustomOp, coralmicro::RegisterCustomOp());

    connect();
    ESP_LOGI(TAG, "Connected as %s", device_id_.c_str());
    vTaskDelay(100 / portTICK_PERIOD_MS);
    subscribeToTopics();
    vTaskDelay(100 / portTICK_PERIOD_MS);
    sendStatus(ClientStatus::STARTED);
    int yieldRc = 0;
    while (!quit_ && yieldRc >= 0)
    {
        yieldRc = mqttClient_->yield(5000);
    }
    ESP_LOGI(TAG, "EdgeBenchClient run completed, yieldRc: %d, quit: %d", yieldRc, quit_);
    disconnect();
}
