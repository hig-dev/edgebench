#ifndef EDGE_BENCH_CLIENT_H
#define EDGE_BENCH_CLIENT_H

#include <string>
#include <vector>
#include "mqtt_topic.h"
#include "mqtt_client.h"
#include "freertos/queue.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

struct MqttMessage {
    std::string     topic;
    std::vector<uint8_t> payload;
};

class EdgeBenchClient {
public:
    EdgeBenchClient(const std::string& device_id,
                    const std::string& broker_host = "127.0.0.1",
                    int broker_port = 1883);
    void connect();
    void disconnect();
    void sendStatus(ClientStatus status);
    void sendResult(int elapsed_time_ms);
    void run();

private:
    static void mqtt_event_handler(void* handler_args,
                                   esp_event_base_t base,
                                   int32_t event_id,
                                   void* event_data);
    void onConnect();
    void onMessage(esp_mqtt_event_handle_t ev);
    void startLatencyTest();
    void startAccuracyTest();

    std::string                          device_id_;
    std::string                          broker_host_;
    int                                  broker_port_;
    Topic                                topic_;
    esp_mqtt_client_handle_t             client_;
    QueueHandle_t                        msg_queue_;
    int                                  iterations_{0};
    Model                                model_{Model::UNKNOWN};
    TestMode                             mode_{TestMode::NONE};
    std::vector<uint8_t>                 latency_input_buffer_;
    std::vector<uint8_t>                 model_buffer_;  
    tflite::MicroInterpreter*            interpreter_{nullptr};
    int8_t*                              input_tensor_{nullptr};
    int8_t*                              output_tensor_{nullptr};
    static constexpr int kArenaSize =    3200 * 1024;
    uint8_t*                             tensor_arena_{nullptr};
    std::string                          intermediate_topic_;
    std::vector<uint8_t>                 intermediate_data_;
};

#endif // EDGE_BENCH_CLIENT_H
