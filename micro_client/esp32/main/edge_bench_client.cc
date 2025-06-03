#include "edge_bench_client.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "esp_event.h"
#include "tensorflow_config.h"

static const char *TAG = "EdgeBenchClient";

EdgeBenchClient::EdgeBenchClient(const std::string &device_id,
                                 const std::string &broker_host,
                                 int broker_port)
    : device_id_(device_id),
      broker_host_(broker_host),
      broker_port_(broker_port),
      topic_(device_id)
{
    esp_mqtt_client_config_t cfg{};
    cfg.broker.address.hostname = broker_host_.c_str();
    cfg.broker.address.port = broker_port_;
    cfg.broker.address.transport = MQTT_TRANSPORT_OVER_TCP;
    cfg.session.protocol_ver = MQTT_PROTOCOL_V_3_1_1;
    cfg.buffer.size = 193 * 1024;
    cfg.buffer.out_size = 66 * 1024;
    client_ = esp_mqtt_client_init(&cfg);
    esp_mqtt_client_register_event(client_,
                                   MQTT_EVENT_ANY,
                                   mqtt_event_handler,
                                   this);
#if I2C_MASTER
    auto i2c_ret = i2c_comm_.init();
    if (i2c_ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C initialization failed: %s", esp_err_to_name(i2c_ret));
        quit_ = true;
    }
#endif
}

void EdgeBenchClient::connect()
{
    ESP_LOGI(TAG, "Connecting to %s:%d", broker_host_.c_str(), broker_port_);
    esp_mqtt_client_start(client_);
}

void EdgeBenchClient::disconnect()
{
    esp_mqtt_client_stop(client_);
    esp_mqtt_client_destroy(client_);
    ESP_LOGI(TAG, "Disconnected from MQTT broker");
}

void EdgeBenchClient::sendStatus(ClientStatus status)
{
    uint8_t b = static_cast<uint8_t>(status);
    esp_mqtt_client_publish(client_,
                            topic_.STATUS().c_str(),
                            reinterpret_cast<const char *>(&b), 1, 1, 0);
    ESP_LOGI(TAG, "Status sent: %d", b);
}

void EdgeBenchClient::sendResult(int elapsed_time_ms)
{
    uint8_t buf[4] = {
        uint8_t(elapsed_time_ms >> 24),
        uint8_t(elapsed_time_ms >> 16),
        uint8_t(elapsed_time_ms >> 8),
        uint8_t(elapsed_time_ms)};
    esp_mqtt_client_publish(client_,
                            topic_.RESULT_LATENCY().c_str(),
                            reinterpret_cast<const char *>(buf), 4, 1, 0);
    ESP_LOGI(TAG, "Result sent: %d ms", elapsed_time_ms);
}

void EdgeBenchClient::startLatencyTest()
{
    ESP_LOGI(TAG, "Running %d iterations...", iterations_);
#if I2C_MASTER
    // Send mode as latency test
    uint8_t mode = static_cast<uint8_t>(TestMode::LATENCY);
    auto ret = i2c_comm_.write(I2CCOMM_FEATURE_MODE, 0, sizeof(mode), &mode);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to set mode: %s", esp_err_to_name(ret));
        quit_ = true;
        return;
    }
    vTaskDelay(100 / portTICK_PERIOD_MS);
    // Send iterations count
    ret = i2c_comm_.write(I2CCOMM_FEATURE_ITERATIONS, 0, sizeof(iterations_), reinterpret_cast<uint8_t *>(&iterations_));
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to set iterations: %s", esp_err_to_name(ret));
        quit_ = true;
        return;
    }
    vTaskDelay(100 / portTICK_PERIOD_MS);
    // Send input tensor
    ESP_LOGI(TAG, "Send input tensor: %zu bytes", model_input_size_);
    ret = i2c_comm_.write(I2CCOMM_FEATURE_INPUT, 0, model_input_size_, reinterpret_cast<uint8_t *>(input_tensor_));
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to send input tensor: %s", esp_err_to_name(ret));
        quit_ = true;
        return;
    }
    // Wait for the device to be ready
    vTaskDelay(1000 / portTICK_PERIOD_MS);
    // Send start command
    uint8_t cmd = static_cast<uint8_t>(Command::START_LATENCY_TEST);
    ret = i2c_comm_.write(I2CCOMM_FEATURE_CMD, 0, sizeof(cmd), &cmd);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to send start command: %s", esp_err_to_name(ret));
        quit_ = true;
        return;
    }
    vTaskDelay(3000 / portTICK_PERIOD_MS);
    // Receive result from I2C device (ms)
    int ms = i2c_comm_.read_latency_result_ms();
    sendStatus(ClientStatus::DONE);
#else
    TickType_t t0 = xTaskGetTickCount();
    for (int i = 0; i < iterations_; ++i)
    {
        auto result = interpreter_->Invoke();
        if (result != kTfLiteOk)
        {
            ESP_LOGE(TAG, "Interpreter Invoke failed: %d", result);
            quit_ = true;
            return;
        }
    }
    TickType_t t1 = xTaskGetTickCount();
    sendStatus(ClientStatus::DONE);
    uint32_t elapsedTicks = (uint32_t)(t1 - t0);
    uint32_t ms = elapsedTicks * portTICK_PERIOD_MS;
#endif
    ESP_LOGI(TAG, "Run completed: %d ms", (int)ms);
    sendResult(ms);
}

void EdgeBenchClient::startAccuracyTest()
{
#if I2C_MASTER
    // Send mode as latency test
    uint8_t mode = static_cast<uint8_t>(TestMode::ACCURACY);
    auto ret = i2c_comm_.write(I2CCOMM_FEATURE_MODE, 0, sizeof(mode), &mode);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to set mode: %s", esp_err_to_name(ret));
        quit_ = true;
        return;
    }
    vTaskDelay(100 / portTICK_PERIOD_MS);
    // Send input tensor
    ESP_LOGI(TAG, "Send input tensor: %zu bytes", model_input_size_);
    ret = i2c_comm_.write(I2CCOMM_FEATURE_INPUT, 0, model_input_size_, reinterpret_cast<uint8_t *>(input_tensor_));
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to send input tensor: %s", esp_err_to_name(ret));
        quit_ = true;
        return;
    }
    // Wait for the device to be ready
    vTaskDelay(1000 / portTICK_PERIOD_MS);
    // Send start command
    uint8_t cmd = static_cast<uint8_t>(Command::START_LATENCY_TEST);
    ret = i2c_comm_.write(I2CCOMM_FEATURE_CMD, 0, sizeof(cmd), &cmd);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to send start command: %s", esp_err_to_name(ret));
        quit_ = true;
        return;
    }
    vTaskDelay(3000 / portTICK_PERIOD_MS);
    // Receive result from I2C device (ms)
    model_output_size_ = DEFAULT_OUTPUT_SIZE;
    auto output_tensor = i2c_comm_.read_accuracy_result(model_output_size_);
    esp_mqtt_client_publish(client_,
                            topic_.RESULT_ACCURACY().c_str(),
                            reinterpret_cast<const char *>(output_tensor.data()),
                            model_output_size_, 1, 0);
    ESP_LOGI(TAG, "Accuracy test result sent, output size: %d bytes", (int)output_tensor.size());
#else
    interpreter_->Invoke();
    esp_mqtt_client_publish(client_,
                            topic_.RESULT_ACCURACY().c_str(),
                            reinterpret_cast<const char *>(output_tensor_),
                            model_output_size_, 1, 0);
    ESP_LOGI(TAG, "Accuracy test result sent, output size: %d bytes", (int)model_output_size_);
#endif
}

void EdgeBenchClient::mqtt_event_handler(void *handler_args,
                                         esp_event_base_t base,
                                         int32_t event_id,
                                         void *event_data)
{
    auto self = static_cast<EdgeBenchClient *>(handler_args);
    auto ev = static_cast<esp_mqtt_event_handle_t>(event_data);
    switch ((esp_mqtt_event_id_t)event_id)
    {
    case MQTT_EVENT_CONNECTED:
        self->onConnect();
        break;
    case MQTT_EVENT_DATA:
        self->onMessage(ev);
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "MQTT_EVENT_ERROR");
        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
        break;
    case MQTT_EVENT_SUBSCRIBED:
        ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED");
        break;
    case MQTT_EVENT_UNSUBSCRIBED:
        ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED");
        break;
    case MQTT_EVENT_PUBLISHED:
        ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED");
        break;
    default:
        break;
    }
}

void EdgeBenchClient::onConnect()
{
    ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
    esp_mqtt_client_subscribe(client_, topic_.CONFIG_MODE().c_str(), 1);
    esp_mqtt_client_subscribe(client_, topic_.CONFIG_ITERATIONS().c_str(), 1);
    esp_mqtt_client_subscribe(client_, topic_.MODEL().c_str(), 1);
    esp_mqtt_client_subscribe(client_, topic_.INPUT_LATENCY().c_str(), 1);
    esp_mqtt_client_subscribe(client_, topic_.INPUT_ACCURACY().c_str(), 1);
    esp_mqtt_client_subscribe(client_, topic_.CMD().c_str(), 1);
}

void EdgeBenchClient::onMessage(esp_mqtt_event_handle_t ev)
{
    if (ev->data_len == ev->total_data_len)
    {
        ESP_LOGI(TAG, "Message on '%.*s' (%zu bytes)",
                 ev->topic_len,
                 ev->topic,
                 ev->data_len);
        auto topic = std::string(ev->topic, ev->topic_len);
        auto data = std::vector<uint8_t>(0);
        if (topic == topic_.INPUT_LATENCY() || topic == topic_.INPUT_ACCURACY())
        {
#if I2C_MASTER
            model_input_size_ = ev->total_data_len;
            if (input_tensor_ == nullptr)
            {
                input_tensor_ = (int8_t *)heap_caps_malloc(
                    model_input_size_,
                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            }
#endif
            if (input_tensor_ == nullptr)
            {
                ESP_LOGE(TAG, "Input tensor not allocated");
                quit_ = true;
                return;
            }
            if (ev->total_data_len != model_input_size_)
            {
                ESP_LOGE(TAG, "Invalid input data size, expected %zu, got %zu",
                         model_input_size_,
                         ev->total_data_len);
                quit_ = true;
                return;
            }
            std::memcpy(input_tensor_, ev->data, model_input_size_);
        }
        else
        {
            data = std::vector<uint8_t>(ev->data, ev->data + ev->data_len);
        }
        handleMessage(topic, data);
        return;
    }
    if (ev->current_data_offset == 0)
    {
        ESP_LOGI(TAG, "Partial message on '%.*s' (%zu bytes)",
                 ev->topic_len,
                 ev->topic,
                 ev->total_data_len);
        auto topic = std::string(ev->topic, ev->topic_len);
        if (topic != topic_.MODEL())
        {
            ESP_LOGE(TAG, "Invalid topic for partial message");
            quit_ = true;
            return;
        }
        if (kModelBufferSize < ev->total_data_len)
        {
            ESP_LOGE(TAG, "Model size too large: %zu bytes", ev->total_data_len);
            quit_ = true;
            return;
        }
        model_size_ = ev->total_data_len;
    }
    if (model_buffer_ == nullptr)
    {
        ESP_LOGE(TAG, "Model buffer not allocated");
        quit_ = true;
        return;
    }
    memcpy(model_buffer_ + ev->current_data_offset,
           ev->data,
           ev->data_len);
    if (ev->current_data_offset + ev->data_len >= ev->total_data_len)
    {
        ESP_LOGI(TAG, "Model load completed (%zu bytes)", ev->total_data_len);
        handleMessage(topic_.MODEL(), std::vector<uint8_t>(0));
    }
}

void EdgeBenchClient::handleMessage(const std::string &topic, const std::vector<uint8_t> &payload)
{
    ESP_LOGI(TAG, "Got message on '%s' (%zu bytes)",
             topic.c_str(),
             payload.size());
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
#if !I2C_MASTER
        if (model_buffer_ == nullptr)
        {
            ESP_LOGE(TAG, "Model buffer is empty");
            quit_ = true;
            return;
        }
        const tflite::Model *model = tflite::GetModel(model_buffer_);
        if (model->version() != TFLITE_SCHEMA_VERSION)
        {
            ESP_LOGE(TAG, "Model schema v%lu not supported", (unsigned long)model->version());
            quit_ = true;
            return;
        }

        auto free_size_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        ESP_LOGI(TAG, "Free PSRAM size: %d", free_size_psram);

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
                                                    kArenaSize);
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
        model_output_size_ = interpreter_->output_tensor(0)->bytes;
#endif
    }
    else if (topic == topic_.INPUT_LATENCY())
    {
        ESP_LOGI(TAG, "Input latency data loaded");
        latency_input_ready_ = true;
    }
    else if (topic == topic_.INPUT_ACCURACY())
    {
        xTaskCreatePinnedToCore(
            [](void *arg)
            {
                auto self = static_cast<EdgeBenchClient *>(arg);
                self->startAccuracyTest();
                vTaskDelete(nullptr);
            },
            "AccuracyTest",
            4 * 1024,
            this,
            1,
            nullptr,
            1);
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
            xTaskCreatePinnedToCore(
                [](void *arg)
                {
                    auto self = static_cast<EdgeBenchClient *>(arg);
                    self->startLatencyTest();
                    vTaskDelete(nullptr);
                },
                "LatencyTest",
                4 * 1024,
                this,
                1,
                nullptr,
                1);
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
            ESP_LOGI(TAG, "State reset");
            return;
        }
    }

    bool latency_config_ready =
        mode_ == TestMode::LATENCY && iterations_ > 0;

    bool accuracy_config_ready =
        mode_ == TestMode::ACCURACY;

    bool config_ready = latency_config_ready || accuracy_config_ready;
#if I2C_MASTER
    bool interpreter_ready = true;
#else
    bool interpreter_ready = interpreter_ != nullptr;
#endif
    bool input_ready = latency_input_ready_ || mode_ == TestMode::ACCURACY;

#if !I2C_MASTER
    if (!sent_ready_for_model_ && config_ready)
    {
        sent_ready_for_model_ = true;
        ESP_LOGI(TAG, "All config received, requesting model");
        sendStatus(ClientStatus::READY_FOR_MODEL);
    }
#endif

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
    connect();
    ESP_LOGI(TAG, "Connected as %s", device_id_.c_str());
    // Wait for the subscriptions to be established
    vTaskDelay(2000 / portTICK_PERIOD_MS);
    sendStatus(ClientStatus::STARTED);
    while (!quit_)
    {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    disconnect();
}
