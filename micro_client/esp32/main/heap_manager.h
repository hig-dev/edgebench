#ifndef HEAP_MANAGER_H
#define HEAP_MANAGER_H

#include <cstdint>

static constexpr int kArenaSize = 3200 * 1024;
static constexpr int kModelBufferSize = 4488760;

// global buffers (defined in .cc)
extern uint8_t* tensor_arena_;
extern uint8_t* model_buffer_;

// reserve functions
void reserve_tensor_arena();
void reserve_model_buffer();

#endif // HEAP_MANAGER_H
