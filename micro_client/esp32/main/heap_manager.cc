#include "heap_manager.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

static const char* TAG = "EdgeBenchClient";

uint8_t* tensor_arena_ = nullptr;
uint8_t* model_buffer_ = nullptr;
tflite::MicroOpResolver* micro_op_resolver_ = nullptr;

bool reserve_tensor_arena() {
    tensor_arena_ = (uint8_t*)heap_caps_malloc(
        kArenaSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!tensor_arena_) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return false;
    }
    ESP_LOGI(TAG, "Tensor arena allocated: %d bytes", kArenaSize);
    return true;
}

bool reserve_model_buffer() {
    model_buffer_ = (uint8_t*)heap_caps_malloc(
        kModelBufferSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!model_buffer_) {
        ESP_LOGE(TAG, "Failed to allocate model buffer");
        return false;
    }
    ESP_LOGI(TAG, "Model buffer allocated: %d bytes", kModelBufferSize);
    return true;
}

void create_op_resolver() {
#if DEIT
    auto *micro_op_resolver = new tflite::MicroMutableOpResolver<19>();
    micro_op_resolver->AddAdd();
    micro_op_resolver->AddBatchMatMul();
    micro_op_resolver->AddConcatenation();
    micro_op_resolver->AddConv2D();
    micro_op_resolver->AddDepthwiseConv2D();
    micro_op_resolver->AddFullyConnected();
    micro_op_resolver->AddGather();
    // GELU is not supported in TensorFlow Lite Micro for ESP32-S3
    //micro_op_resolver->AddGelu();
    micro_op_resolver->AddMean();
    micro_op_resolver->AddMul();
    micro_op_resolver->AddPad();
    micro_op_resolver->AddReshape();
    micro_op_resolver->AddResizeNearestNeighbor();
    micro_op_resolver->AddRsqrt();
    micro_op_resolver->AddSoftmax();
    micro_op_resolver->AddSquaredDifference();
    micro_op_resolver->AddStridedSlice();
    micro_op_resolver->AddSub();
    micro_op_resolver->AddTranspose();
    micro_op_resolver_ = micro_op_resolver;
#elif EFFICIENTVIT
    auto *micro_op_resolver = new tflite::MicroMutableOpResolver<17>();
    micro_op_resolver->AddAdd();
    micro_op_resolver->AddBatchMatMul();
    micro_op_resolver->AddConcatenation();
    micro_op_resolver->AddConv2D();
    micro_op_resolver->AddDepthwiseConv2D();
    micro_op_resolver->AddDequantize();
    micro_op_resolver->AddDiv();
    micro_op_resolver->AddHardSwish();
    micro_op_resolver->AddMul();
    micro_op_resolver->AddPad();
    micro_op_resolver->AddPadV2();
    micro_op_resolver->AddQuantize();
    micro_op_resolver->AddRelu();
    micro_op_resolver->AddReshape();
    micro_op_resolver->AddResizeNearestNeighbor();
    micro_op_resolver->AddStridedSlice();
    micro_op_resolver->AddTranspose();
    micro_op_resolver_ = micro_op_resolver;
#else // MOBILEONE
    auto *micro_op_resolver = new tflite::MicroMutableOpResolver<8>();
    micro_op_resolver->AddAdd();
    micro_op_resolver->AddConv2D();
    micro_op_resolver->AddDepthwiseConv2D();
    micro_op_resolver->AddMul();
    micro_op_resolver->AddPad();
    micro_op_resolver->AddResizeNearestNeighbor();
    micro_op_resolver->AddLogistic();
    micro_op_resolver->AddMean();
    micro_op_resolver_ = micro_op_resolver;
#endif
}