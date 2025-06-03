#include "tensorflow_config.h"

#if !CORALMICRO
uint8_t* tensor_arena_ = nullptr;
uint8_t* model_buffer_ = nullptr;
#endif

tflite::MicroOpResolver* micro_op_resolver_ = nullptr;

#if EDGETPU
std::shared_ptr<coralmicro::EdgeTpuContext> tpu_context_ = nullptr;
#endif

bool reserve_tensor_arena() {
    #if ESP32
    tensor_arena_ = (uint8_t*)heap_caps_malloc(
        kArenaSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!tensor_arena_) {
        return false;
    }
    return true;
    #elif CORALMICRO
    return true; // Static allocation in SDRAM
    #endif
    return false;
}

bool reserve_model_buffer() {
    #if ESP32
    model_buffer_ = (uint8_t*)heap_caps_malloc(
        kModelBufferSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!model_buffer_) {
        return false;
    }
    return true;
    #elif CORALMICRO
    return true; // Static allocation in SDRAM
    #endif
    return false;
}

void create_op_resolver() {
#if I2C_MASTER
    // Op resolver not needed for I2C master mode
#elif EDGETPU
    auto *micro_op_resolver = new tflite::MicroMutableOpResolver<1>();
    micro_op_resolver->AddCustom(coralmicro::kCustomOp, coralmicro::RegisterCustomOp());    
    micro_op_resolver_ = micro_op_resolver;
#elif DEIT
    auto *micro_op_resolver = new tflite::MicroMutableOpResolver<19>();
    micro_op_resolver->AddAdd();
    micro_op_resolver->AddBatchMatMul(); // Not available for coralmicro
    micro_op_resolver->AddConcatenation();
    micro_op_resolver->AddConv2D();
    micro_op_resolver->AddDepthwiseConv2D();
    micro_op_resolver->AddFullyConnected();
    micro_op_resolver->AddGather();
    micro_op_resolver->AddGelu(); // Not available for coralmicro and esp32
    micro_op_resolver->AddMean();
    micro_op_resolver->AddMul();
    micro_op_resolver->AddPad();
    micro_op_resolver->AddReshape();
    micro_op_resolver->AddResizeNearestNeighbor();
    micro_op_resolver->AddRsqrt();
    micro_op_resolver->AddSoftmax();
    micro_op_resolver->AddSquaredDifference(); // Not available for coralmicro
    micro_op_resolver->AddStridedSlice();
    micro_op_resolver->AddSub();
    micro_op_resolver->AddTranspose();
    micro_op_resolver_ = micro_op_resolver;
#elif EFFICIENTVIT
    auto *micro_op_resolver = new tflite::MicroMutableOpResolver<17>();
    micro_op_resolver->AddAdd();
    micro_op_resolver->AddBatchMatMul(); // Not available for coralmicro
    micro_op_resolver->AddConcatenation();
    micro_op_resolver->AddConv2D();
    micro_op_resolver->AddDepthwiseConv2D();
    micro_op_resolver->AddDequantize();
    micro_op_resolver->AddDiv(); // Not available for coralmicro
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

TensorflowConfigResult initialize_tensorflow_config() {
    #if EDGETPU
    tpu_context_ = coralmicro::EdgeTpuManager::GetSingleton()->OpenDevice(coralmicro::PerformanceMode::kMax);
    if (!tpu_context_) {
        return TensorflowConfigResult::EDGE_TPU_INIT_FAILED;
    }
    #endif
    if (!reserve_tensor_arena()) {
        return TensorflowConfigResult::TENSOR_ARENA_ALLOC_FAILED;
    }

    if (!reserve_model_buffer()) {
        return TensorflowConfigResult::MODEL_BUFFER_ALLOC_FAILED;
    }
    create_op_resolver();
    return TensorflowConfigResult::SUCCESS;
}