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
#include "mqtt.h"
#include "../shared/mqtt_topic.h"

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

  // auto client  = EdgeBenchClient("coralmicro",
  //                                MQTT_BROKER_HOST,
  //                                MQTT_BROKER_PORT);
  // client.run();

  bool mqttConnectSuccess = connectToMqttBroker(MQTT_BROKER_HOST, MQTT_BROKER_PORT);
  if (!mqttConnectSuccess)
  {
    printf("Failed to connect to MQTT broker\r\n");
    return;
  }
  auto topic = Topic("coralmicro");
  subscribeToMqttTopic(topic.CONFIG_ITERATIONS().c_str(), MQTTQoS1);
  uint8_t payload[4] = "123";
  publishMqttMessage(topic.CONFIG_ITERATIONS().c_str(), payload, 4, MQTTQoS1);
  processMqttLoop(nullptr);
}
