#include "edge_bench_client.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "esp_event.h"
#include <chrono>

static const char* TAG = "EdgeBenchClient";

EdgeBenchClient::EdgeBenchClient(const std::string& device_id,
                                 const std::string& broker_host,
                                 int broker_port)
  : device_id_(device_id),
    broker_host_(broker_host),
    broker_port_(broker_port),
    topic_(device_id)
{
    msg_queue_ = xQueueCreate(5, sizeof(MqttMessage*));
    esp_mqtt_client_config_t cfg{};
    cfg.broker.address.hostname  = broker_host_.c_str();
    cfg.broker.address.port      = broker_port_;
    cfg.broker.address.transport = MQTT_TRANSPORT_OVER_TCP;
    cfg.session.protocol_ver = MQTT_PROTOCOL_V_3_1_1;
    cfg.buffer.size = 16 * 1024;
    cfg.buffer.out_size = 16 * 1024;
    client_ = esp_mqtt_client_init(&cfg);
    esp_mqtt_client_register_event(client_,
                                   MQTT_EVENT_ANY,
                                   mqtt_event_handler,
                                   this);
}

void EdgeBenchClient::connect() {
    ESP_LOGI(TAG, "Connecting to %s:%d", broker_host_.c_str(), broker_port_);
    esp_mqtt_client_start(client_);
}

void EdgeBenchClient::disconnect() {
    esp_mqtt_client_stop(client_);
    esp_mqtt_client_destroy(client_);
    ESP_LOGI(TAG, "Disconnected from MQTT broker");
}

void EdgeBenchClient::sendStatus(ClientStatus status) {
    uint8_t b = static_cast<uint8_t>(status);
    esp_mqtt_client_publish(client_,
                            topic_.STATUS().c_str(),
                            reinterpret_cast<const char*>(&b), 1, 1, 0);
    ESP_LOGI(TAG, "Status sent: %d", b);
}

void EdgeBenchClient::sendResult(int elapsed_time_ms) {
    uint8_t buf[4] = {
        uint8_t(elapsed_time_ms >> 24),
        uint8_t(elapsed_time_ms >> 16),
        uint8_t(elapsed_time_ms >>  8),
        uint8_t(elapsed_time_ms)
    };
    esp_mqtt_client_publish(client_,
                            topic_.RESULT_LATENCY().c_str(),
                            reinterpret_cast<const char*>(buf), 4, 1, 0);
    ESP_LOGI(TAG, "Result sent: %d ms", elapsed_time_ms);
}

void EdgeBenchClient::startLatencyTest() {
    ESP_LOGI(TAG, "Running %d iterations...", iterations_);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations_; ++i) {
        interpreter_->Invoke();
    }
    auto t1 = std::chrono::steady_clock::now();
    sendStatus(ClientStatus::DONE);
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    ESP_LOGI(TAG, "Run completed: %d ms", ms);
    sendResult(ms);
}

void EdgeBenchClient::startAccuracyTest() {
    interpreter_->Invoke();
    size_t out_bytes = interpreter_->output_tensor(0)->bytes;
    esp_mqtt_client_publish(client_,
                            topic_.RESULT_ACCURACY().c_str(),
                            reinterpret_cast<const char*>(output_tensor_),
                            out_bytes, 1, 0);
    ESP_LOGI(TAG, "Accuracy result sent");
}

void EdgeBenchClient::mqtt_event_handler(void* handler_args,
                                         esp_event_base_t base,
                                         int32_t event_id,
                                         void* event_data)
{
    auto self = static_cast<EdgeBenchClient*>(handler_args);
    auto ev   = static_cast<esp_mqtt_event_handle_t>(event_data);
    switch ((esp_mqtt_event_id_t)event_id) {
      case MQTT_EVENT_CONNECTED: self->onConnect(); break;
      case MQTT_EVENT_DATA:      self->onMessage(ev); break;
      case MQTT_EVENT_ERROR:     ESP_LOGE(TAG, "MQTT_EVENT_ERROR"); break;
      case MQTT_EVENT_DISCONNECTED: ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED"); break;
      case MQTT_EVENT_SUBSCRIBED: ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED"); break;
      case MQTT_EVENT_UNSUBSCRIBED: ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED"); break;
      case MQTT_EVENT_PUBLISHED: ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED"); break;
      default: break;
    }
}

void EdgeBenchClient::onConnect() {
    ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
    esp_mqtt_client_subscribe(client_, topic_.CONFIG_MODE().c_str(),       1);
    esp_mqtt_client_subscribe(client_, topic_.CONFIG_ITERATIONS().c_str(), 1);
    esp_mqtt_client_subscribe(client_, topic_.CONFIG_MODEL().c_str(),      1);
    esp_mqtt_client_subscribe(client_, topic_.MODEL().c_str(),             1);
    esp_mqtt_client_subscribe(client_, topic_.INPUT_LATENCY().c_str(),     1);
    esp_mqtt_client_subscribe(client_, topic_.INPUT_ACCURACY().c_str(),    1);
    esp_mqtt_client_subscribe(client_, topic_.CMD().c_str(),               1);
}

void EdgeBenchClient::onMessage(esp_mqtt_event_handle_t ev) {
    if (ev->current_data_offset == 0) {
        ESP_LOGI(TAG, "Received on %.*s", ev->topic_len, ev->topic);
        ESP_LOGI(TAG, "Data length: %d", ev->data_len);
        ESP_LOGI(TAG, "Total data length: %d", ev->total_data_len);
        ESP_LOGI(TAG, "Data offset: %d", ev->current_data_offset);
        intermediate_data_.clear();
        intermediate_data_.resize(ev->total_data_len);
        intermediate_topic_.assign(ev->topic, ev->topic_len);
    }
    std::memcpy(intermediate_data_.data() + ev->current_data_offset,
                ev->data,
                ev->data_len);
    if (ev->current_data_offset + ev->data_len >= ev->total_data_len) {
        ESP_LOGI(TAG, "Message complete on '%s' (%zu bytes)",
                     intermediate_topic_.c_str(),
                     intermediate_data_.size());
        auto *msg = new MqttMessage{
            std::move(intermediate_topic_),
            std::move(intermediate_data_)
        };
        intermediate_data_ = std::vector<uint8_t>(0);
        intermediate_topic_ = std::string();
        if (xQueueSend(msg_queue_, &msg, 0) != pdPASS) {
            delete msg;
            ESP_LOGE(TAG, "Failed to send message to queue");
        }
    }
}

void EdgeBenchClient::run() {
    connect();
    ESP_LOGI(TAG, "Connected as %s", device_id_.c_str());
    // Wait for the subscriptions to be established
    vTaskDelay(2000 / portTICK_PERIOD_MS);
    sendStatus(ClientStatus::STARTED);

    bool sent_ready_for_model = false;
    bool sent_ready_for_task  = false;
    MqttMessage *msg;
    ESP_LOGI(TAG, "Waiting for message...");
    while (xQueueReceive(msg_queue_, &msg, portMAX_DELAY) == pdPASS) {
        ESP_LOGI(TAG, "Got message on '%s' (%zu bytes)",
                     msg->topic.c_str(),
                     msg->payload.size());
        auto t = msg->topic;
        if (t == topic_.CONFIG_MODE()) {
            if (msg->payload.size() != 1) {
                ESP_LOGE(TAG, "Invalid mode message");
                break;
            }
            mode_ = static_cast<TestMode>(msg->payload[0]);
            ESP_LOGI(TAG, "Mode set: %d", int(mode_));
        }
        else if (t == topic_.CONFIG_ITERATIONS()) {
            if (msg->payload.size() != 4) {
                ESP_LOGE(TAG, "Invalid iterations message");
                break;
            }
            // Assume big-endian 4-byte integer
            iterations_ =
                (static_cast<int>(msg->payload[0]) << 24) |
                (static_cast<int>(msg->payload[1]) << 16) |
                (static_cast<int>(msg->payload[2]) <<  8) |
                static_cast<int>(msg->payload[3]);
            ESP_LOGI(TAG, "Iterations set: %d", iterations_);
        }
        else if (t == topic_.CONFIG_MODEL()) {
            if (msg->payload.size() != 4) {
                ESP_LOGE(TAG, "Invalid model message");
                break;
            }
            // Assume big-endian 4-byte integer
            auto model_asint =
                (static_cast<int>(msg->payload[0]) << 24) |
                (static_cast<int>(msg->payload[1]) << 16) |
                (static_cast<int>(msg->payload[2]) <<  8) |
                static_cast<int>(msg->payload[3]);
            model_ = static_cast<Model>(model_asint);
            ESP_LOGI(TAG, "Model set: %d", int(model_));
        }
        else if (t == topic_.MODEL()) {
            model_buffer_ = std::move(msg->payload);
            const tflite::Model* model = tflite::GetModel(model_buffer_.data());
            if (model->version() != TFLITE_SCHEMA_VERSION) {
                ESP_LOGE(TAG, "Model schema v%lu not supported", (unsigned long)model->version());
                break;
            }

            auto free_size_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            ESP_LOGI(TAG, "Free PSRAM size: %d", free_size_psram);

            if (tensor_arena_ == nullptr) {
                ESP_LOGI(TAG, "Allocating tensor arena: %d bytes", kArenaSize);
                tensor_arena_ = (uint8_t *) heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            }
            if (tensor_arena_ == nullptr) {
                ESP_LOGE(TAG, "Failed to allocate tensor arena");
                break;
            }

            delete interpreter_;
#if DEIT
            auto micro_op_resolver = tflite::MicroMutableOpResolver<19>();
            micro_op_resolver.AddAdd();
            micro_op_resolver.AddBatchMatMul();
            micro_op_resolver.AddConcatenation();
            micro_op_resolver.AddConv2D();
            micro_op_resolver.AddDepthwiseConv2D();
            micro_op_resolver.AddFullyConnected();
            micro_op_resolver.AddGather();
            // GELU is not supported in TensorFlow Lite Micro for ESP32-S3
            //micro_op_resolver.AddGelu();
            micro_op_resolver.AddMean();
            micro_op_resolver.AddMul();
            micro_op_resolver.AddPad();
            micro_op_resolver.AddReshape();
            micro_op_resolver.AddResizeNearestNeighbor();
            micro_op_resolver.AddRsqrt();
            micro_op_resolver.AddSoftmax();
            micro_op_resolver.AddSquaredDifference();
            micro_op_resolver.AddStridedSlice();
            micro_op_resolver.AddSub();
            micro_op_resolver.AddTranspose();
#elif EFFICIENTVIT
            auto micro_op_resolver = tflite::MicroMutableOpResolver<17>();
            micro_op_resolver.AddAdd();
            micro_op_resolver.AddBatchMatMul();
            micro_op_resolver.AddConcatenation();
            micro_op_resolver.AddConv2D();
            micro_op_resolver.AddDepthwiseConv2D();
            micro_op_resolver.AddDequantize();
            micro_op_resolver.AddDiv();
            micro_op_resolver.AddHardSwish();
            micro_op_resolver.AddMul();
            micro_op_resolver.AddPad();
            micro_op_resolver.AddPadV2();
            micro_op_resolver.AddQuantize();
            micro_op_resolver.AddRelu();
            micro_op_resolver.AddReshape();
            micro_op_resolver.AddResizeNearestNeighbor();
            micro_op_resolver.AddStridedSlice();
            micro_op_resolver.AddTranspose();
#else // MOBILEONE
            auto micro_op_resolver = tflite::MicroMutableOpResolver<8>();
            micro_op_resolver.AddAdd();
            micro_op_resolver.AddConv2D();
            micro_op_resolver.AddDepthwiseConv2D();
            micro_op_resolver.AddMul();
            micro_op_resolver.AddPad();
            micro_op_resolver.AddResizeNearestNeighbor();
            micro_op_resolver.AddLogistic();
            micro_op_resolver.AddMean();
#endif

            interpreter_ = new tflite::MicroInterpreter(model,
                                                        micro_op_resolver,
                                                        tensor_arena_,
                                                        kArenaSize);
            TfLiteStatus allocate_status = interpreter_->AllocateTensors();
            if (allocate_status != kTfLiteOk) {
                ESP_LOGE(TAG, "AllocateTensors() failed");
                break;
            }
            input_tensor_  = interpreter_->typed_input_tensor<int8_t>(0);
            output_tensor_ = interpreter_->typed_output_tensor<int8_t>(0);
            ESP_LOGI(TAG, "Model loaded; tensors allocated");

            if (mode_ == TestMode::LATENCY) {
                ESP_LOGI(TAG, "Copy input data to tensor");
                size_t input_size = interpreter_->input_tensor(0)->bytes;
                if (latency_input_buffer_.size() != input_size) {
                    ESP_LOGE(TAG, "Invalid input data size, expected %zu, got %zu",
                            input_size,
                            latency_input_buffer_.size());
                    break;
                }
                std::memcpy(input_tensor_, latency_input_buffer_.data(), input_size);
                latency_input_buffer_.clear();
                latency_input_buffer_.resize(0);
            }
        }
        else if (t == topic_.INPUT_LATENCY()) {
            latency_input_buffer_ = std::move(msg->payload);
            ESP_LOGI(TAG, "Input latency data loaded");
        }
        else if (t == topic_.INPUT_ACCURACY()) {
            // run single inference and publish output buffer
            size_t input_size = interpreter_->input_tensor(0)->bytes;
            if (msg->payload.size() != input_size) {
                ESP_LOGE(TAG, "Invalid input data size, expected %zu, got %zu",
                         input_size,
                         msg->payload.size());
                break;
            }
            std::memcpy(input_tensor_, msg->payload.data(), input_size);
            xTaskCreatePinnedToCore(
                [](void* arg) {
                    auto self = static_cast<EdgeBenchClient*>(arg);
                    self->startAccuracyTest();
                    vTaskDelete(nullptr);
                },
                "AccuracyTest",
                8 * 1024,
                this,
                1,
                nullptr,
                1);
        }
        else if (t == topic_.CMD()) {
            if (msg->payload.size() != 1) {
                ESP_LOGE(TAG, "Invalid command message");
                break;
            }
            Command cmd = static_cast<Command>(msg->payload[0]);
            if (cmd == Command::START_LATENCY_TEST) {
                xTaskCreatePinnedToCore(
                    [](void* arg) {
                        auto self = static_cast<EdgeBenchClient*>(arg);
                        self->startLatencyTest();
                        vTaskDelete(nullptr);
                    },
                    "LatencyTest",
                    8 * 1024,
                    this,
                    1,
                    nullptr,
                    1);
            } else if (cmd == Command::STOP) {
                disconnect();
                break;
            } else if (cmd == Command::RESET) {
                sent_ready_for_model = false;
                sent_ready_for_task  = false;
                iterations_ = 0;
                latency_input_buffer_.clear();
                latency_input_buffer_.resize(0);
                model_buffer_.clear();
                model_buffer_.resize(0);
                mode_       = TestMode::NONE;
                model_      = Model::UNKNOWN;
                delete interpreter_;
                interpreter_ = nullptr;
                delete msg;
                ESP_LOGI(TAG, "State reset");
                continue;
            }
        }

        bool latency_config_ready = 
            mode_ == TestMode::LATENCY
            && model_ != Model::UNKNOWN
            && iterations_ > 0 
            && (latency_input_buffer_.size() > 0 || sent_ready_for_model);
        
        bool accuracy_config_ready = 
            mode_ == TestMode::ACCURACY
            && model_ != Model::UNKNOWN;

        bool config_ready = latency_config_ready || accuracy_config_ready;
        bool interpreter_ready = interpreter_ != nullptr;

        if (!sent_ready_for_model && config_ready) {
            sent_ready_for_model = true;
            ESP_LOGI(TAG, "All config received, requesting model");
            sendStatus(ClientStatus::READY_FOR_MODEL);
        }
        
        if (!sent_ready_for_task && config_ready && interpreter_ready) {
            sent_ready_for_task = true;
            ESP_LOGI(TAG, "Ready for task, waiting for input or command");
            sendStatus(ClientStatus::READY_FOR_TASK);
        }
        delete msg;
    }
    delete msg;
}
