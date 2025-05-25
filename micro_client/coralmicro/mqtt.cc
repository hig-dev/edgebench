#include "mqtt.h"
#include <errno.h>
#include <string.h>
#include "coralmicro/libs/base/network.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"


// Helper: get current time in ms
static uint32_t prvGetTimeMs( void )
{
    return xTaskGetTickCount() * portTICK_PERIOD_MS;
}

// Helper: MQTT event callback
static void prvEventCallback( MQTTContext_t * pxMQTTContext,
                              MQTTPacketInfo_t * pxPacketInfo,
                              MQTTDeserializedInfo_t * pxDeserializedInfo )
{
    ( void ) pxMQTTContext;
    if( ( pxPacketInfo->type & 0xF0U ) == MQTT_PACKET_TYPE_CONNACK )
    {
        printf("MQTT CONNACK received\r\n");
    }
    else {
        printf("MQTT packet received: type=0x%02X, remainingLength=%zu\r\n",
               pxPacketInfo->type, pxPacketInfo->remainingLength);
    }
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
bool connectToMqttBroker(const char* broker_url, int port) {
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
                   prvEventCallback,
                   &mqttBuffer ) != MQTTSuccess )
    {
        printf("MQTT_Init failed\r\n");
        return false;
    }

    printf("MQTT context info:\r\n");
    printf("  Connect status: %d\r\n", mqttContext.connectStatus);

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