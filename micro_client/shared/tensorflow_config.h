#ifndef TENSORFLOW_CONFIG_H
#define TENSORFLOW_CONFIG_H

#include <cstdint>

#if ESP32
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "esp_heap_caps.h"
#elif CORALMICRO
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "../coralmicro/coralmicro/libs/tensorflow/utils.h"
#endif

#if EDGETPU
#include "../coralmicro/coralmicro/libs/tpu/edgetpu_op.h"
#endif

#if I2C_MASTER
static constexpr int kModelBufferSize = 1;
static constexpr int kArenaSize = 1;
#elif ESP32
static constexpr int kModelBufferSize = 4488760;
static constexpr int kArenaSize = 3241 * 1024;
#elif CORALMICRO
static constexpr int kModelBufferSize = 16040576;
static constexpr int kArenaSize = 3403 * 1024;
#endif

#if !CORALMICRO
extern uint8_t* tensor_arena_;
extern uint8_t* model_buffer_;
#endif

extern tflite::MicroOpResolver* micro_op_resolver_;
#if EDGETPU
extern std::shared_ptr<coralmicro::EdgeTpuContext> tpu_context_;
#endif

enum class TensorflowConfigResult : int {
    SUCCESS     = 0,
    TENSOR_ARENA_ALLOC_FAILED = 1,
    MODEL_BUFFER_ALLOC_FAILED = 2,
    EDGE_TPU_INIT_FAILED = 3,
};

TensorflowConfigResult initialize_tensorflow_config();

#endif