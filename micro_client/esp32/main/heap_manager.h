#ifndef HEAP_MANAGER_H
#define HEAP_MANAGER_H

#include <cstdint>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define EFFICIENTVIT 0
#define DEIT 0

#if EFFICIENTVIT
static constexpr int kArenaSize = 3241 * 1024;
static constexpr int kModelBufferSize = 1189512;
#elif DEIT
static constexpr int kArenaSize = 3200 * 1024;
static constexpr int kModelBufferSize = 6377096;
#else // MOBILEONE
static constexpr int kArenaSize = 3200 * 1024;
static constexpr int kModelBufferSize = 4488760;
#endif

// global buffers (defined in .cc)
extern uint8_t* tensor_arena_;
extern uint8_t* model_buffer_;
extern tflite::MicroOpResolver* micro_op_resolver_;

// reserve functions
bool reserve_tensor_arena();
bool reserve_model_buffer();
void create_op_resolver();

#endif // HEAP_MANAGER_H
