// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "coralmicro/libs/base/tasks.h"
#include "coralmicro/libs/base/wifi.h"
#include "../shared/wifi_config.h"
#include "../shared/mqtt_config.h"
#include "../shared/mqtt_topic.h"
#include "MQTTClient.h"
#include "coralipstack.h"
#include "coraltimer.h"

void messageArrived(MQTT::MessageData &md)
{
  MQTT::Message &message = md.message;

  printf("Message: qos %d, retained %d, dup %d, packetid %d\r\n",
         message.qos, message.retained, message.dup, message.id);
  // printf("Payload %.*s\n", message.payloadlen, (char *)message.payload);
}

extern "C" void app_main(void *param)
{
  printf("Attempting to use Wi-Fi...\r\n");
  bool wifiTurnOnSuccess = coralmicro::WiFiTurnOn(true);
  if (!wifiTurnOnSuccess)
  {
    printf("Failed to turn on Wi-Fi\r\n");
    return;
  }
  coralmicro::WiFiSetDefaultSsid(WIFI_SSID);
  coralmicro::WiFiSetDefaultPsk(WIFI_PASS);
  bool wifiConnectSuccess = coralmicro::WiFiConnect();
  if (!wifiConnectSuccess)
  {
    printf("Failed to connect to Wi-Fi\r\n");
    return;
  }
  printf("Wi-Fi connected\r\n");

  int sockfd = coralmicro::SocketClient(MQTT_BROKER_HOST, MQTT_BROKER_PORT);
  if (sockfd < 0)
  {
    printf("SocketClient failed to connect to %s:%d\r\n", MQTT_BROKER_HOST, MQTT_BROKER_PORT);
    return;
  }
  printf("SocketClient connected to %s:%d\r\n", MQTT_BROKER_HOST, MQTT_BROKER_PORT);

  // auto client  = EdgeBenchClient("coralmicro",
  //                                MQTT_BROKER_HOST,
  //                                MQTT_BROKER_PORT);
  // client.run();

  auto ipstack = CoralIPStack(sockfd);
  auto socketFd = ipstack.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT);
  if (socketFd < 0)
  {
    printf("Failed to connect to MQTT broker\r\n");
    return;
  }
  auto client = MQTT::Client<CoralIPStack, CoralTimer>(ipstack);
  char topic[] = "test";

  MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
  data.MQTTVersion = 4;
  data.clientID.cstring = (char *)"coralmicro";
  auto rc = client.connect(data);
  if (rc != 0)
    printf("rc from MQTT connect is %d\n", rc);
  printf("MQTT connected\r\n");

  rc = client.subscribe(topic, MQTT::QOS1, messageArrived);
  if (rc != 0)
    printf("rc from MQTT subscribe is %d\n", rc);
  printf("Subscribed to %s\r\n", topic);

  // char buf[4] = {0x01, 0x02, 0x03, 0x04};
  // MQTT::Message message;
  // message.qos = MQTT::QOS1;
  // message.retained = false;
  // message.dup = false;
  // message.payload = (void *)buf;
  // message.payloadlen = 4;
  // rc = client.publish(topic, message);
  // if (rc != 0)
  //   printf("rc from MQTT publish is %d\r\n", rc);
  // printf("MQTT published status\r\n");

  int counter = 0;
  while (true)
  {
    // if (counter % 100 == 0)
    // {
    //   printf("Counter: %d\r\n", counter);
    //     char buf2[4] = {0x01, 0x03, 0x03, 0x04};
    //     MQTT::Message message2;
    //     message2.qos = MQTT::QOS1;
    //     message2.retained = false;
    //     message2.dup = false;
    //     message2.payload = (void *)buf2;
    //     message2.payloadlen = 4;
    //   rc = client.publish(topic, message2);
    //   if (rc != 0)
    //     printf("rc from MQTT publish is %d\r\n", rc);
    //   printf("MQTT published status\r\n");
    // }
    client.yield(100);
    counter++;
  }
}
