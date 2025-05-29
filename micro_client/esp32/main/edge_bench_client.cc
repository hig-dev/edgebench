#include "edge_bench_client.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "esp_event.h"
#include "heap_manager.h"

static const char* TAG = "EdgeBenchClient";

EdgeBenchClient::EdgeBenchClient(const std::string& device_id,
                                 const std::string& broker_host,
                                 int broker_port)
  : device_id_(device_id),
    broker_host_(broker_host),
    broker_port_(broker_port),
    topic_(device_id)
{
    esp_mqtt_client_config_t cfg{};
    cfg.broker.address.hostname  = broker_host_.c_str();
    cfg.broker.address.port      = broker_port_;
    cfg.broker.address.transport = MQTT_TRANSPORT_OVER_TCP;
    cfg.session.protocol_ver = MQTT_PROTOCOL_V_3_1_1;
    cfg.buffer.size = 193 * 1024;
    cfg.buffer.out_size = 193 * 1024;
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
    auto t0 = xTaskGetTickCount() * portTICK_PERIOD_MS;
    for (int i = 0; i < iterations_; ++i) {
        interpreter_->Invoke();
    }
    auto t1 = xTaskGetTickCount() * portTICK_PERIOD_MS;
    sendStatus(ClientStatus::DONE);
    int ms = t1 - t0;
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
    esp_mqtt_client_subscribe(client_, topic_.MODEL().c_str(),             1);
    esp_mqtt_client_subscribe(client_, topic_.INPUT_LATENCY().c_str(),     1);
    esp_mqtt_client_subscribe(client_, topic_.INPUT_ACCURACY().c_str(),    1);
    esp_mqtt_client_subscribe(client_, topic_.CMD().c_str(),               1);
}

void EdgeBenchClient::onMessage(esp_mqtt_event_handle_t ev) {
    if (ev->data_len == ev->total_data_len) {
        ESP_LOGI(TAG, "Message on '%.*s' (%zu bytes)",
                     ev->topic_len,
                     ev->topic,
                     ev->data_len);
        auto topic = std::string(ev->topic, ev->topic_len);
        auto data = std::vector<uint8_t>(0);
        if (topic == topic_.INPUT_LATENCY() || topic == topic_.INPUT_ACCURACY()) {
            if (input_tensor_ == nullptr) {
                ESP_LOGE(TAG, "Input tensor not allocated");
                quit_ = true;
                return;
            }
            if (ev->total_data_len != model_input_size_) {
                ESP_LOGE(TAG, "Invalid input data size, expected %zu, got %zu",
                        model_input_size_,
                        ev->total_data_len);
                quit_ = true;
                return;
            }
            std::memcpy(input_tensor_, ev->data, model_input_size_);
        }
        else {
            data = std::vector<uint8_t>(ev->data, ev->data + ev->data_len);
        }
        handleMessage(topic, data);
        return;
    }
    if (ev->current_data_offset == 0) {
        ESP_LOGI(TAG, "Partial message on '%.*s' (%zu bytes)",
                     ev->topic_len,
                     ev->topic,
                     ev->total_data_len);
        auto topic = std::string(ev->topic, ev->topic_len);
        if (topic != topic_.MODEL()) {
            ESP_LOGE(TAG, "Invalid topic for partial message");
            quit_ = true;
            return;
        }
        if (kModelBufferSize < ev->total_data_len) {
            ESP_LOGE(TAG, "Model size too large: %zu bytes", ev->total_data_len);
            quit_ = true;
            return;
        }
        model_size_ = ev->total_data_len;
    }
    if (model_buffer_ == nullptr) {
        ESP_LOGE(TAG, "Model buffer not allocated");
        quit_ = true;
        return;
    }
    memcpy(model_buffer_ + ev->current_data_offset,
           ev->data,
           ev->data_len);
    if (ev->current_data_offset + ev->data_len >= ev->total_data_len) {
        ESP_LOGI(TAG, "Model load completed (%zu bytes)", ev->total_data_len);
        handleMessage(topic_.MODEL(), std::vector<uint8_t>(0));
    }
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
        if (model_buffer_ == nullptr) {
            ESP_LOGE(TAG, "Model buffer is empty");
            quit_ = true;
            return;
        }
        const tflite::Model* model = tflite::GetModel(model_buffer_);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE(TAG, "Model schema v%lu not supported", (unsigned long)model->version());
            quit_ = true;
            return;
        }

        auto free_size_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        ESP_LOGI(TAG, "Free PSRAM size: %d", free_size_psram);

        delete interpreter_;

        if (tensor_arena_ == nullptr) {
            ESP_LOGE(TAG, "Tensor arena not allocated");
            quit_ = true;
            return;
        }
        if (micro_op_resolver_ == nullptr) {
            ESP_LOGE(TAG, "Micro op resolver not allocated");
            quit_ = true;
            return;
        }
        interpreter_ = new tflite::MicroInterpreter(model,
                                                    *micro_op_resolver_,
                                                    tensor_arena_,
                                                    kArenaSize);
        TfLiteStatus allocate_status = interpreter_->AllocateTensors();
        if (allocate_status != kTfLiteOk) {
            ESP_LOGE(TAG, "AllocateTensors() failed");
            quit_ = true;
            return;
        }
        input_tensor_  = interpreter_->typed_input_tensor<int8_t>(0);
        output_tensor_ = interpreter_->typed_output_tensor<int8_t>(0);
        ESP_LOGI(TAG, "Model loaded; tensors allocated");ESP_LOGI(TAG, "Copy input data to tensor");
        model_input_size_ = interpreter_->input_tensor(0)->bytes;
    }
    else if (topic == topic_.INPUT_LATENCY()) {
        ESP_LOGI(TAG, "Input latency data loaded");
        latency_input_ready_ = true;
    }
    else if (topic == topic_.INPUT_ACCURACY()) {
        xTaskCreatePinnedToCore(
            [](void* arg) {
                auto self = static_cast<EdgeBenchClient*>(arg);
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
    else if (topic == topic_.CMD()) {
        if (payload.size() != 1) {
            ESP_LOGE(TAG, "Invalid command message");
            quit_ = true;
            return;
        }
        Command cmd = static_cast<Command>(payload[0]);
        if (cmd == Command::START_LATENCY_TEST) {
            xTaskCreatePinnedToCore(
                [](void* arg) {
                    auto self = static_cast<EdgeBenchClient*>(arg);
                    self->startLatencyTest();
                    vTaskDelete(nullptr);
                },
                "LatencyTest",
                4 * 1024,
                this,
                1,
                nullptr,
                1);
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
            delete interpreter_;
            interpreter_ = nullptr;
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
    bool interpreter_ready = interpreter_ != nullptr;
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
    connect();
    ESP_LOGI(TAG, "Connected as %s", device_id_.c_str());
    // Wait for the subscriptions to be established
    vTaskDelay(2000 / portTICK_PERIOD_MS);
    sendStatus(ClientStatus::STARTED);
    while (!quit_) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    disconnect();
}
