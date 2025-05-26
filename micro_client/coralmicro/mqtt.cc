#include "mqtt.h"
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include "coralmicro/libs/base/network.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

struct NetworkContext
{
    int socketFd;
};

static NetworkContext networkContext = {-1};

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
    if ( status == coralmicro::IOStatus::kOk )
    {
        printf("Sent %d bytes to socket %d\r\n", (int)bytesToSend, fd);
        return ( int32_t ) bytesToSend;
    }
    else
    {
        printf("Failed to send %d bytes to socket %d: %s (%d)\r\n",
               (int)bytesToSend, fd, strerror(errno), errno);
        return -1; // Indicate error
    }
}

// Transport receive using coralmicro network API
static int32_t transportRecv( NetworkContext_t * pNetworkContext,
                              void * pBuffer,
                              size_t bytesToRecv )
{
    int fd = pNetworkContext->socketFd;
    auto status = coralmicro::ReadBytes( fd, pBuffer, bytesToRecv );
    if( status == coralmicro::IOStatus::kOk ) {
        printf("Received %d bytes from socket %d\r\n", (int)bytesToRecv, fd);
        return ( int32_t ) bytesToRecv;
    } else if( status == coralmicro::IOStatus::kEof ) {
        printf("Socket %d closed by peer\r\n", fd);
        return 0;
    }
    printf("Failed to receive %d bytes from socket %d: %s (%d)\r\n",
           (int)bytesToRecv, fd, strerror(errno), errno);
    return -1;
}

static void mqtt_event_handler(MQTTContext_t * pxMQTTContext,
                              MQTTPacketInfo_t * pxPacketInfo,
                              MQTTDeserializedInfo_t * pxDeserializedInfo)
{
    ( void ) pxMQTTContext;
    printf("MQTT event handler called \r\n");
    if( ( pxPacketInfo->type & 0xF0U ) == MQTT_PACKET_TYPE_PUBLISH )
    {
        // Handle incoming PUBLISH messages
        printf("PUBLISH received for packet id %u.\n\n",
                   pxDeserializedInfo->packetIdentifier );
        MQTTPublishInfo_t * pxPublishInfo = pxDeserializedInfo->pPublishInfo;
        printf("Topic: %.*s\n", pxPublishInfo->topicNameLength, pxPublishInfo->pTopicName );
        printf("Payload length: %zu\n", pxPublishInfo->payloadLength );
        // Invoke callback to process the incoming publish
        // std::vector<uint8_t> payload(
        //     static_cast<const uint8_t*>(pxPublishInfo->pPayload),
        //     static_cast<const uint8_t*>(pxPublishInfo->pPayload) + pxPublishInfo->payloadLength);
        // std::string topic(pxPublishInfo->pTopicName, pxPublishInfo->topicNameLength);
        // if (edgeBenchClientInstance_ == nullptr) {
        //     ESP_LOGE(TAG, "EdgeBenchClient instance is null");
        //     return;
        // }
        // edgeBenchClientInstance_->handleMessage(topic, payload);
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


// Connect to MQTT broker
bool connectToMqttBroker(const char* broker_url, int port) {
    printf("Connecting to MQTT broker at %s:%d...\r\n", broker_url, port);

    // Establish TCP connection via coralmicro network API
    int sockfd = coralmicro::SocketClient( broker_url, port );
    if( sockfd < 0 ) {
        printf( "SocketClient failed to connect to %s:%d\r\n", broker_url, port );
        return false;
    }
    networkContext = {
        .socketFd = sockfd
    };

    static TransportInterface_t mqttTransport = TransportInterface_t{
        .recv = ( TransportRecv_t ) transportRecv,
        .send = ( TransportSend_t ) transportSend,
        .pNetworkContext = &networkContext,
    };

    // Initialize MQTT
    auto initStatus = MQTT_Init( &mqttContext,
                   &mqttTransport,
                   prvGetTimeMs,
                   mqtt_event_handler,
                   &mqttBuffer ) ;
    if (initStatus != MQTTSuccess) {
        printf("MQTT_Init failed with status %d\r\n", initStatus);
        return false;
    }
    printf("MQTT_Init successful\r\n");

    // MQTT Connect info
    static MQTTConnectInfo_t connectInfo = MQTTConnectInfo_t{
        .cleanSession = true,
        .keepAliveSeconds = MQTT_KEEP_ALIVE_SECONDS,
        .pClientIdentifier = MQTT_CLIENT_IDENTIFIER,
        .clientIdentifierLength = (uint16_t)strlen(MQTT_CLIENT_IDENTIFIER),
        .pUserName = NULL,
        .userNameLength = 0,
        .pPassword = NULL,
        .passwordLength = 0,
    };

    static bool sessionPresent = false;

    // Send CONNECT
    auto connectStatus = MQTT_Connect( &mqttContext,
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
    printf("Disconnecting from MQTT broker...\r\n");

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

void processMqttLoop(void * pvParameters)
{
    printf("Starting MQTT_ProcessLoop...\r\n");
    bool continueProcessing = true;
    while(continueProcessing)
    {
        auto processLoopStatus = MQTT_ProcessLoop(&mqttContext, 100);
        continueProcessing = processLoopStatus == MQTTSuccess;
        //vTaskDelay(pdMS_TO_TICKS(50));
        if (!continueProcessing)
        {
            printf("MQTT_ProcessLoop failed with status %d\r\n", processLoopStatus);
        }
    }
}