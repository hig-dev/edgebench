#ifndef MICR0_CLIENT_MQTT_H
#define MICR0_CLIENT_MQTT_H

#include "core_mqtt.h"
#include "core_mqtt_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// MQTT configuration macros
#define MQTT_CLIENT_IDENTIFIER       "coraledgebench"
#define MQTT_BUFFER_SIZE              8*1024
#define MQTT_MAX_PUBLISH_RECORDS      15
#define MQTT_KEEP_ALIVE_SECONDS       60
#define MQTT_CONNECT_TIMEOUT_MS       1000
#define MQTT_TRANSPORT_TIMEOUT_MS     200
#define MQTT_PROCESS_LOOP_TIMEOUT_MS  2000 

// MQTT network context

// Global MQTT objects
static uint8_t mqttSharedBuffer[ MQTT_BUFFER_SIZE ];
static MQTTFixedBuffer_t mqttBuffer = { mqttSharedBuffer, MQTT_BUFFER_SIZE };
static MQTTContext_t mqttContext;


/**
 * @brief Connect to MQTT broker.
 *
 * @param broker_url The URL or IP address of the MQTT broker.
 * @param port The network port of the MQTT broker.
 * @return true if the connection was successful, false otherwise.
 */
bool connectToMqttBroker(const char* broker_url, int port);

/**
 * @brief Disconnect from the MQTT broker.
 * @return true if the disconnection was successful, false otherwise.
 */
bool disconnectFromMqttBroker();

/**
 * @brief Publish a message to an MQTT topic.
 *
 * @param topic The topic to publish to.
 * @param payload The message payload.
 * @param payload_length The length of the payload.
 * @param qos The Quality of Service level for the publish.
 * @return true if the publish was successful, false otherwise.
 */
bool publishMqttMessage(const char* topic, const uint8_t* payload, size_t payload_length, MQTTQoS_t qos);

/**
 * @brief Subscribe to an MQTT topic.
 *
 * @param topic The topic to subscribe to.
 * @param qos The Quality of Service level for the subscription.
 * @return true if the subscription was successful, false otherwise.
 */
bool subscribeToMqttTopic(const char* topic, MQTTQoS_t qos);

/**
 * @brief Process the MQTT loop, handling incoming messages and sending keep-alive packets.
 *
 * This function should be called periodically to maintain the MQTT connection.
 */
void processMqttLoop(void * pvParameters);


#ifdef __cplusplus
}
#endif

#endif // MICR0_CLIENT_MQTT_H