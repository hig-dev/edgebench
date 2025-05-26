#ifndef MQTT_TOPIC_H
#define MQTT_TOPIC_H

#include <string>

enum class TestMode : int {
    NONE     = 0,
    LATENCY = 1,
    ACCURACY = 2
};

enum class ClientStatus : int {
    NONE    = 0,
    STARTED = 1,
    READY_FOR_MODEL = 2,
    READY_FOR_INPUT = 3,
    READY_FOR_TASK  = 4,
    DONE    = 5
};

enum class Command : int {
    NONE              = 0,
    START_LATENCY_TEST = 1,
    STOP               = 2,
    RESET              = 3
};

class Topic {
public:
    explicit Topic(const std::string& device_id)
      : device_id_(device_id) {}

    std::string CONFIG_ITERATIONS() const {
        return "bench/" + device_id_ + "/config/iterations";
    }

    std::string CONFIG_MODE() const {
        return "bench/" + device_id_ + "/config/mode";
    }

    std::string MODEL() const {
        return "bench/" + device_id_ + "/model";
    }

    std::string INPUT_LATENCY() const {
        return "bench/" + device_id_ + "/input/latency";
    }

    std::string INPUT_ACCURACY() const {
        return "bench/" + device_id_ + "/input/accuracy";
    }

    std::string CMD() const {
        return "bench/" + device_id_ + "/cmd";
    }

    std::string STATUS() const {
        return "bench/" + device_id_ + "/status";
    }

    std::string RESULT_LATENCY() const {
        return "bench/" + device_id_ + "/result/latency";
    }

    std::string RESULT_ACCURACY() const {
        return "bench/" + device_id_ + "/result/accuracy";
    }

private:
    std::string device_id_;
};

#endif // MQTT_TOPIC_H
