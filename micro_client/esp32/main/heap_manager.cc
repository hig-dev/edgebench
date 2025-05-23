#include "heap_manager.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

static const char* TAG = "EdgeBenchClient";

uint8_t* tensor_arena_ = nullptr;
uint8_t* model_buffer_ = nullptr;

void reserve_tensor_arena() {
    tensor_arena_ = (uint8_t*)heap_caps_malloc(
        kArenaSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!tensor_arena_) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return;
    }
    ESP_LOGI(TAG, "Tensor arena allocated: %d bytes", kArenaSize);
}

void reserve_model_buffer() {
    model_buffer_ = (uint8_t*)heap_caps_malloc(
        kModelBufferSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!model_buffer_) {
        ESP_LOGE(TAG, "Failed to allocate model buffer");
        return;
    }
    ESP_LOGI(TAG, "Model buffer allocated: %d bytes", kModelBufferSize);
}

