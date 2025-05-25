#ifndef MICR0_CLIENT_MQTT_H
#define MICR0_CLIENT_MQTT_H

#include "core_mqtt.h"
#include "core_mqtt_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// MQTT configuration macros
#define MQTT_CLIENT_IDENTIFIER       "coraledgebench"
#define MQTT_BUFFER_SIZE              1024
#define MQTT_MAX_PUBLISH_RECORDS      15
#define MQTT_KEEP_ALIVE_SECONDS       60
#define MQTT_CONNECT_TIMEOUT_MS       1000
#define MQTT_TRANSPORT_TIMEOUT_MS     200

struct NetworkContext
{
    int socketFd;
};

// MQTT network context
static NetworkContext networkContext;

// Global MQTT objects
static uint8_t mqttSharedBuffer[ MQTT_BUFFER_SIZE ];
static MQTTFixedBuffer_t mqttBuffer = { mqttSharedBuffer, MQTT_BUFFER_SIZE };
static MQTTContext_t mqttContext;
static TransportInterface_t mqttTransport;
static MQTTPubAckInfo_t outgoingPublishRecords[ MQTT_MAX_PUBLISH_RECORDS ];
static MQTTPubAckInfo_t incomingPublishRecords[ MQTT_MAX_PUBLISH_RECORDS ];

/**
 * @brief Connect to MQTT broker.
 *
 * @param broker_url The URL or IP address of the MQTT broker.
 * @param port The network port of the MQTT broker.
 * @return true if the connection was successful, false otherwise.
 */
bool connectToMqttBroker(const char* broker_url, int port);

#ifdef __cplusplus
}
#endif

#endif // MICR0_CLIENT_MQTT_H