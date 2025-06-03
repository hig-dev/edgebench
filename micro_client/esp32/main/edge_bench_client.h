#ifndef EDGE_BENCH_CLIENT_H
#define EDGE_BENCH_CLIENT_H

#define I2C_MASTER 1

#include <string>
#include <vector>
#include "mqtt_topic.h"
#include "mqtt_client.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#if I2C_MASTER
#include "i2c_comm.h"
#endif

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
    void handleMessage(const std::string &topic, const std::vector<uint8_t> &payload);
    void startLatencyTest();
    void startAccuracyTest();

    std::string                          device_id_;
    std::string                          broker_host_;
    int                                  broker_port_;
    Topic                                topic_;
    esp_mqtt_client_handle_t             client_;
    int                                  iterations_{0};
    TestMode                             mode_{TestMode::NONE};
    int                                  model_size_{0};
    size_t                               model_input_size_{0};
    size_t                               model_output_size_{0};
    bool                                 latency_input_ready_{false};
    tflite::MicroInterpreter*            interpreter_{nullptr};
    int8_t*                              input_tensor_{nullptr};
    int8_t*                              output_tensor_{nullptr};
    bool                                 sent_ready_for_model_{false};
    bool                                 sent_ready_for_input_{false};
    bool                                 sent_ready_for_task_{false};
    bool                                 quit_{false};
    #if I2C_MASTER
    I2CComm                              i2c_comm_{DEFAULT_I2C_ADDRESS};
    #endif
};

#endif // EDGE_BENCH_CLIENT_H
