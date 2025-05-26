#include "mqtt.h"
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include "coralmicro/libs/base/network.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"


// Helper: get current time in ms
static uint32_t prvGetTimeMs( void )
{
    return xTaskGetTickCount() * portTICK_PERIOD_MS;
}

// Transport send using coralmicro network API
static int32_t transportSend( NetworkContext_t * pNetworkContext,
                              const void * pBuffer,
                              size_t bytesToSend )
{
    int fd = pNetworkContext->socketFd;
    auto status = coralmicro::WriteBytes( fd, pBuffer, bytesToSend );
    return ( status == coralmicro::IOStatus::kOk ) ? ( int32_t ) bytesToSend : -1;
}

// Transport receive using coralmicro network API
static int32_t transportRecv( NetworkContext_t * pNetworkContext,
                              void * pBuffer,
                              size_t bytesToRecv )
{
    int fd = pNetworkContext->socketFd;
    auto status = coralmicro::ReadBytes( fd, pBuffer, bytesToRecv );
    if( status == coralmicro::IOStatus::kOk ) {
        return ( int32_t ) bytesToRecv;
    } else if( status == coralmicro::IOStatus::kEof ) {
        return 0;
    }
    return -1;
}

// Connect to MQTT broker
bool connectToMqttBroker(const char* broker_url, int port, MQTTEventCallback_t userCallback) {
    printf("Connecting to MQTT broker at %s:%d...\r\n", broker_url, port);

    // Establish TCP connection via coralmicro network API
    int sockfd = coralmicro::SocketClient( broker_url, port );
    if( sockfd < 0 ) {
        printf( "SocketClient failed to connect to %s:%d\r\n", broker_url, port );
        return false;
    }
    networkContext.socketFd = sockfd;

    // Setup transport interface
    mqttTransport.pNetworkContext = &networkContext;
    mqttTransport.send = ( TransportSend_t ) transportSend;
    mqttTransport.recv = ( TransportRecv_t ) transportRecv;

    // Initialize MQTT
    if( MQTT_Init( &mqttContext,
                   &mqttTransport,
                   prvGetTimeMs,
                   userCallback,
                   &mqttBuffer ) != MQTTSuccess )
    {
        printf("MQTT_Init failed\r\n");
        return false;
    }

    if (MQTT_InitStatefulQoS(&mqttContext,
                        outgoingPublishRecords,
                        MQTT_MAX_PUBLISH_RECORDS,
                        incomingPublishRecords,
                        MQTT_MAX_PUBLISH_RECORDS) != MQTTSuccess)
    {
        printf("MQTT_InitStatefulQoS failed\r\n");
        return false;
    }

    // MQTT Connect info
    auto connectInfo = MQTTConnectInfo_t{
        .cleanSession = true,
        .keepAliveSeconds = MQTT_KEEP_ALIVE_SECONDS,
        .pClientIdentifier = MQTT_CLIENT_IDENTIFIER,
        .clientIdentifierLength = (uint16_t)strlen(MQTT_CLIENT_IDENTIFIER),
        .pUserName = NULL,
        .userNameLength = 0,
        .pPassword = NULL, // Replace with your MQTT password
        .passwordLength = 0,
    };

    bool sessionPresent = false;

    // Send CONNECT
    MQTTStatus_t connectStatus = MQTT_Connect( &mqttContext,
                      &connectInfo,
                      NULL,
                      MQTT_CONNECT_TIMEOUT_MS,
                      &sessionPresent );
    if(connectStatus != MQTTSuccess )
    {
        printf("MQTT_Connect failed with status %d\r\n", connectStatus);
        return false;
    }

    printf("Successfully connected to MQTT broker\r\n");
    return true;
}

// Disconnect from MQTT broker
bool disconnectFromMqttBroker()
{
    printf("Disconnecting from MQTT broker at socket %d...\r\n", networkContext.socketFd);

    /* Send MQTT DISCONNECT */
    MQTTStatus_t mqttStatus = MQTT_Disconnect(&mqttContext);
    if (mqttStatus != MQTTSuccess)
    {
        printf("MQTT_Disconnect failed with status %d\r\n", mqttStatus);
    }

    /* Close the TCP socket */
    if ( close(networkContext.socketFd) != 0 )
    {
        printf("Socket close failed: %s (%d)\r\n", strerror(errno), errno);
        return false;
    }

    return ( mqttStatus == MQTTSuccess );
}

// Subscribe to an MQTT topic
bool subscribeToMqttTopic(const char* topic, MQTTQoS_t qos)
{
    MQTTSubscribeInfo_t subscribeInfo[] = {
        {
            .qos = qos,
            .pTopicFilter = topic,
            .topicFilterLength = (uint16_t)strlen(topic)
        }
    };

    uint16_t packetId = MQTT_GetPacketId(&mqttContext);
    MQTTStatus_t status = MQTT_Subscribe(&mqttContext, subscribeInfo, 1, packetId);

    if (status != MQTTSuccess)
    {
        printf("MQTT_Subscribe failed for topic %s with status %d\r\n", topic, status);
        return false;
    }

    printf("MQTT_Subscribe sent for topic %s (packet id %u)\r\n", topic, packetId);

    // Process the MQTT loop to ensure subscription is acknowledged
    MQTTStatus_t loopStatus = processMqttLoopWithTimeout(MQTT_PROCESS_LOOP_TIMEOUT_MS);
    if (loopStatus != MQTTSuccess)
    {
        printf("MQTT_ProcessLoop failed after subscribing to topic %s with status %d\r\n", topic, loopStatus);
        return false;
    }
    return true;
}

bool publishMqttMessage (const char* topic, const uint8_t* payload, size_t payload_length, MQTTQoS_t qos)
{
    MQTTPublishInfo_t publishInfo = {
        .qos = qos,
        .retain = false,
        .dup = false,
        .pTopicName = topic,
        .topicNameLength = (uint16_t)strlen(topic),
        .pPayload = payload,
        .payloadLength = payload_length
    };

    uint16_t packetId = MQTT_GetPacketId(&mqttContext);
    MQTTStatus_t status = MQTT_Publish(&mqttContext, &publishInfo, packetId);

    if (status != MQTTSuccess)
    {
        printf("MQTT_Publish failed for topic %s with status %d\r\n", topic, status);
        return false;
    }

    printf("MQTT_Publish sent for topic %s (packet id %u)\r\n", topic, packetId);
    return true;
}

MQTTStatus_t processMqttLoopWithTimeout(uint32_t ulTimeoutMs )
{
    uint32_t ulMqttProcessLoopTimeoutTime;
    uint32_t ulCurrentTime;

    MQTTStatus_t eMqttStatus = MQTTSuccess;

    ulCurrentTime = mqttContext.getTime();
    ulMqttProcessLoopTimeoutTime = ulCurrentTime + ulTimeoutMs;

    /* Call MQTT_ProcessLoop multiple times a timeout happens, or
     * MQTT_ProcessLoop fails. */
    while( ( ulCurrentTime < ulMqttProcessLoopTimeoutTime ) &&
           ( eMqttStatus == MQTTSuccess || eMqttStatus == MQTTNeedMoreBytes ) )
    {
        printf("Calling MQTT_ProcessLoop...\r\n");
        eMqttStatus = MQTT_ProcessLoop(&mqttContext);
        ulCurrentTime = mqttContext.getTime();
    }

    if( eMqttStatus == MQTTNeedMoreBytes )
    {
        eMqttStatus = MQTTSuccess;
    }

    return eMqttStatus;
}