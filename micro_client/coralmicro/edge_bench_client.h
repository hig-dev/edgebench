#ifndef EDGE_BENCH_CLIENT_H
#define EDGE_BENCH_CLIENT_H

#include <string>
#include <vector>
#include "../shared/mqtt_topic.h"
#include "MQTTClient.h"
#include "coralipstack.h"
#include "coraltimer.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"

static constexpr int kMqttMaxPayloadSize = 196808;

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
    void handleMessage(const std::string &topic, const std::vector<uint8_t> &payload);

private:
    void subscribeToTopics();
    void startLatencyTest();
    void startAccuracyTest();
    int publishMQTTMessage(const std::string &topic,
                                 const uint8_t *payload,
                                 size_t payload_len,
                                 MQTT::QoS qos = MQTT::QOS1);
    std::string                          device_id_;
    std::string                          broker_host_;
    int                                  broker_port_;
    Topic                                topic_;
    CoralIPStack                         ipstack_{};
    MQTT::Client<CoralIPStack, CoralTimer, kMqttMaxPayloadSize, 6>* mqttClient_{nullptr};
    int                                  iterations_{0};
    TestMode                             mode_{TestMode::NONE};
    size_t                               model_input_size_{0};
    bool                                 latency_input_ready_{false};
    tflite::MicroErrorReporter           error_reporter_{};
    tflite::MicroInterpreter*            interpreter_{nullptr};
    int8_t*                              input_tensor_{nullptr};
    int8_t*                              output_tensor_{nullptr};
    bool                                 sent_ready_for_model_{false};
    bool                                 sent_ready_for_input_{false};
    bool                                 sent_ready_for_task_{false};
    bool                                 quit_{false};
};

#endif // EDGE_BENCH_CLIENT_H
