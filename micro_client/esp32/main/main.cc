/* Blink Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"

static const char *TAG = "example";

/* Use project configuration menu (idf.py menuconfig) to choose the GPIO to blink,
   or you can edit the following line and set a number here.
*/
const gpio_num_t BLINK_GPIO = GPIO_NUM_21;

[[noreturn]] void blink_task(void* param) {
  bool on = true;
  while (true) {
    on = !on;
    gpio_set_level(BLINK_GPIO, on);
    vTaskDelay(pdMS_TO_TICKS(500));
  }
}

static void configure_led(void)
{
    ESP_LOGI(TAG, "Example configured to blink GPIO LED!");
    gpio_reset_pin(BLINK_GPIO);
    /* Set the GPIO as a push/pull output */
    gpio_set_direction(BLINK_GPIO, GPIO_MODE_OUTPUT);
}


extern "C" void app_main(void* param)
{
    /* Configure the peripheral according to the LED type */
    configure_led();

    xTaskCreate(&blink_task, "blink_user_led_task", 4*1024, nullptr, 5, nullptr);
    vTaskSuspend(nullptr);
}