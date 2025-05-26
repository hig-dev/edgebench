#include "edge_bench_client.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
//#include "heap_manager.h"
#include <chrono>
#include <stdio.h>
#include <stdarg.h>

static const char* TAG = "EdgeBenchClient";

void ESP_LOGI(const char* tag, const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("[%s] ", tag);
    vfprintf(stdout, format, args);
    printf("\r\n");
    va_end(args);
}

void ESP_LOGE(const char* tag, const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("[%s] ERROR: ", tag);
    vfprintf(stderr, format, args);
    printf("\r\n");
    va_end(args);
}

static void mqtt_event_handler(MQTTContext_t * pxMQTTContext,
                              MQTTPacketInfo_t * pxPacketInfo,
                              MQTTDeserializedInfo_t * pxDeserializedInfo)
{
    ( void ) pxMQTTContext;
    if( ( pxPacketInfo->type & 0xF0U ) == MQTT_PACKET_TYPE_PUBLISH )
    {
        // Handle incoming PUBLISH messages
        printf("PUBLISH received for packet id %u.\n\n",
                   pxDeserializedInfo->packetIdentifier );
        MQTTPublishInfo_t * pxPublishInfo = pxDeserializedInfo->pPublishInfo;
        printf("Topic: %.*s\n", pxPublishInfo->topicNameLength, pxPublishInfo->pTopicName );
        printf("Payload length: %zu\n", pxPublishInfo->payloadLength );
        // Invoke callback to process the incoming publish
        std::vector<uint8_t> payload(
            static_cast<const uint8_t*>(pxPublishInfo->pPayload),
            static_cast<const uint8_t*>(pxPublishInfo->pPayload) + pxPublishInfo->payloadLength);
        std::string topic(pxPublishInfo->pTopicName, pxPublishInfo->topicNameLength);
        if (edgeBenchClientInstance_ == nullptr) {
            ESP_LOGE(TAG, "EdgeBenchClient instance is null");
            return;
        }
        edgeBenchClientInstance_->handleMessage(topic, payload);
    }
    else
    {
        // Handle ACKs
        switch ( pxPacketInfo->type )
        {
            case MQTT_PACKET_TYPE_PUBACK:
                printf("PUBACK received for packet ID %u.\r\n", pxDeserializedInfo->packetIdentifier );
                break;

            case MQTT_PACKET_TYPE_SUBACK:
                printf("SUBACK received for packet ID %u.\r\n", pxDeserializedInfo->packetIdentifier );
                break;
            default:
                printf("Received unhandled packet type: %02X\r\n", pxPacketInfo->type );
                break;
        }
    }
}

EdgeBenchClient::EdgeBenchClient(const std::string& device_id,
                                 const std::string& broker_host,
                                 int broker_port)
  : device_id_(device_id),
    broker_host_(broker_host),
    broker_port_(broker_port),
    topic_(device_id)
{
    edgeBenchClientInstance_ = this;
}

void EdgeBenchClient::connect() {
    ESP_LOGI(TAG, "Connecting to %s:%d", broker_host_.c_str(), broker_port_);
    bool mqttConnectSuccess = connectToMqttBroker(broker_host_.c_str(), broker_port_, mqtt_event_handler);
    if (!mqttConnectSuccess) {
        ESP_LOGE(TAG, "Failed to connect to MQTT broker");
        return;
    }
    onConnect();
}

void EdgeBenchClient::disconnect() {
    bool mqttDisconnectSuccess = disconnectFromMqttBroker();
    if (!mqttDisconnectSuccess) {
        ESP_LOGE(TAG, "Failed to disconnect from MQTT broker");
        return;
    }
    ESP_LOGI(TAG, "Disconnected from MQTT broker");
}

void EdgeBenchClient::sendStatus(ClientStatus status) {
    uint8_t b = static_cast<uint8_t>(status);
    publishMqttMessage(topic_.STATUS().c_str(), &b, 1, MQTTQoS1);
    ESP_LOGI(TAG, "Status sent: %d", b);
}

void EdgeBenchClient::sendResult(int elapsed_time_ms) {
    uint8_t buf[4] = {
        uint8_t(elapsed_time_ms >> 24),
        uint8_t(elapsed_time_ms >> 16),
        uint8_t(elapsed_time_ms >>  8),
        uint8_t(elapsed_time_ms)
    };
    publishMqttMessage(topic_.RESULT_LATENCY().c_str(), buf, sizeof(buf), MQTTQoS1);
    ESP_LOGI(TAG, "Result sent: %d ms", elapsed_time_ms);
}

void EdgeBenchClient::startLatencyTest() {
    ESP_LOGI(TAG, "Running %d iterations...", iterations_);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations_; ++i) {
        //interpreter_->Invoke();
    }
    auto t1 = std::chrono::steady_clock::now();
    sendStatus(ClientStatus::DONE);
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    ESP_LOGI(TAG, "Run completed: %d ms", ms);
    sendResult(ms);
}

void EdgeBenchClient::startAccuracyTest() {
    //interpreter_->Invoke();
    //size_t out_bytes = interpreter_->output_tensor(0)->bytes;
    size_t out_bytes  = 0;
    publishMqttMessage(topic_.RESULT_LATENCY().c_str(),
                            reinterpret_cast<uint8_t*>(output_tensor_),
                            out_bytes, MQTTQoS1);
    ESP_LOGI(TAG, "Accuracy result sent");
}

void EdgeBenchClient::onConnect() {
    ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
    
    processMqttLoopWithTimeout(MQTT_PROCESS_LOOP_TIMEOUT_MS);
    
    subscribeToMqttTopic(topic_.CONFIG_MODE().c_str(),       MQTTQoS1);
    subscribeToMqttTopic(topic_.CONFIG_ITERATIONS().c_str(), MQTTQoS1);
    subscribeToMqttTopic(topic_.MODEL().c_str(),             MQTTQoS1);
    subscribeToMqttTopic(topic_.INPUT_LATENCY().c_str(),     MQTTQoS1);
    subscribeToMqttTopic(topic_.INPUT_ACCURACY().c_str(),    MQTTQoS1);
    subscribeToMqttTopic(topic_.CMD().c_str(),               MQTTQoS1);

    ESP_LOGI(TAG, "Subscribed to topics");
}

void EdgeBenchClient::handleMessage(const std::string& topic, const std::vector<uint8_t>& payload) {
    ESP_LOGI(TAG, "Got message on '%s' (%zu bytes)",
                     topic.c_str(),
                     payload.size());
    if (topic == topic_.CONFIG_MODE()) {
        if (payload.size() != 1) {
            ESP_LOGE(TAG, "Invalid mode message");
            quit_ = true;
            return;
        }
        mode_ = static_cast<TestMode>(payload[0]);
        ESP_LOGI(TAG, "Mode set: %d", int(mode_));
    }
    else if (topic == topic_.CONFIG_ITERATIONS()) {
        if (payload.size() != 4) {
            ESP_LOGE(TAG, "Invalid iterations message");
            quit_ = true;
            return;
        }
        // Assume big-endian 4-byte integer
        iterations_ =
            (static_cast<int>(payload[0]) << 24) |
            (static_cast<int>(payload[1]) << 16) |
            (static_cast<int>(payload[2]) <<  8) |
            static_cast<int>(payload[3]);
        ESP_LOGI(TAG, "Iterations set: %d", iterations_);
    }
    else if (topic == topic_.MODEL()) {
        ESP_LOGE(TAG, "Model data received, but model loading is not implemented in this example");
        // if (model_buffer_ == nullptr) {
        //     ESP_LOGE(TAG, "Model buffer is empty");
        //     quit_ = true;
        //     return;
        // }
        // const tflite::Model* model = tflite::GetModel(model_buffer_);
        // if (model->version() != TFLITE_SCHEMA_VERSION) {
        //     ESP_LOGE(TAG, "Model schema v%lu not supported", (unsigned long)model->version());
        //     quit_ = true;
        //     return;
        // }

        // auto free_size_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        // ESP_LOGI(TAG, "Free PSRAM size: %d", free_size_psram);

        // delete interpreter_;

        // if (tensor_arena_ == nullptr) {
        //     ESP_LOGE(TAG, "Tensor arena not allocated");
        //     quit_ = true;
        //     return;
        // }
        // if (micro_op_resolver_ == nullptr) {
        //     ESP_LOGE(TAG, "Micro op resolver not allocated");
        //     quit_ = true;
        //     return;
        // }
        // interpreter_ = new tflite::MicroInterpreter(model,
        //                                             *micro_op_resolver_,
        //                                             tensor_arena_,
        //                                             kArenaSize);
        // TfLiteStatus allocate_status = interpreter_->AllocateTensors();
        // if (allocate_status != kTfLiteOk) {
        //     ESP_LOGE(TAG, "AllocateTensors() failed");
        //     quit_ = true;
        //     return;
        // }
        // input_tensor_  = interpreter_->typed_input_tensor<int8_t>(0);
        // output_tensor_ = interpreter_->typed_output_tensor<int8_t>(0);
        // ESP_LOGI(TAG, "Model loaded; tensors allocated");ESP_LOGI(TAG, "Copy input data to tensor");
        // model_input_size_ = interpreter_->input_tensor(0)->bytes;
    }
    else if (topic == topic_.INPUT_LATENCY()) {
        ESP_LOGI(TAG, "Input latency data loaded");
        latency_input_ready_ = true;
    }
    else if (topic == topic_.INPUT_ACCURACY()) {
        xTaskCreate(
            [](void* arg) {
                auto self = static_cast<EdgeBenchClient*>(arg);
                self->startAccuracyTest();
                vTaskDelete(nullptr);
            },
            "AccuracyTest",
            4 * 1024,
            this,
            1,
            nullptr);
    }
    else if (topic == topic_.CMD()) {
        if (payload.size() != 1) {
            ESP_LOGE(TAG, "Invalid command message");
            quit_ = true;
            return;
        }
        Command cmd = static_cast<Command>(payload[0]);
        if (cmd == Command::START_LATENCY_TEST) {
            xTaskCreate(
                [](void* arg) {
                    auto self = static_cast<EdgeBenchClient*>(arg);
                    self->startLatencyTest();
                    vTaskDelete(nullptr);
                },
                "LatencyTest",
                4 * 1024,
                this,
                1,
                nullptr);
        } else if (cmd == Command::STOP) {
            quit_ = true;
            return;
        } else if (cmd == Command::RESET) {
            sent_ready_for_model_ = false;
            sent_ready_for_input_ = false;
            sent_ready_for_task_  = false;
            iterations_ = 0;
            latency_input_ready_ = false;
            mode_       = TestMode::NONE;
            //delete interpreter_;
            //interpreter_ = nullptr;
            ESP_LOGI(TAG, "State reset");
            return;
        }
    }

    bool latency_config_ready = 
        mode_ == TestMode::LATENCY
        && iterations_ > 0;
    
    bool accuracy_config_ready = 
        mode_ == TestMode::ACCURACY;

    bool config_ready = latency_config_ready || accuracy_config_ready;
    bool interpreter_ready = false; //interpreter_ != nullptr;
    bool input_ready = latency_input_ready_ || mode_ == TestMode::ACCURACY;

    if (!sent_ready_for_model_ && config_ready) {
        sent_ready_for_model_ = true;
        ESP_LOGI(TAG, "All config received, requesting model");
        sendStatus(ClientStatus::READY_FOR_MODEL);
    }

    if (!sent_ready_for_input_ && config_ready && interpreter_ready && mode_ == TestMode::LATENCY) {
        sent_ready_for_input_ = true;
        ESP_LOGI(TAG, "Interpreter ready, waiting for input");
        sendStatus(ClientStatus::READY_FOR_INPUT);
    }
    
    if (!sent_ready_for_task_ && config_ready && interpreter_ready && input_ready) {
        sent_ready_for_task_ = true;
        ESP_LOGI(TAG, "Ready for task, waiting for input or command");
        sendStatus(ClientStatus::READY_FOR_TASK);
    }
}

void EdgeBenchClient::run() {
    ESP_LOGI(TAG, "Starting EdgeBenchClient for device %s", device_id_.c_str());
    connect();
    ESP_LOGI(TAG, "Connected as %s", device_id_.c_str());
    // Wait for the subscriptions to be established
    vTaskDelay(2000 / portTICK_PERIOD_MS);
    sendStatus(ClientStatus::STARTED);
    while (!quit_) {
        processMqttLoopWithTimeout(MQTT_PROCESS_LOOP_TIMEOUT_MS);
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
    disconnect();
}
