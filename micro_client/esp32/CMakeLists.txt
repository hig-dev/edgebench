# For more information about build system see
# https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html
# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

#set(COMPONENTS main) Uncommented to use PSRAM

add_compile_definitions(
  ESP32=1
  I2C_MASTER=0
  MOBILEONE=1
  EFFICIENTVIT=0
  DEIT=0
)

# NOTES on EfficientViT:
# EfficientViT models with ESP-NN optimizations fail at runtime:
# tflite::(anonymous namespace)::Eval(TfLiteContext*, TfLiteNode*) at /home/hig/src/edgebench/micro_client/esp32/managed_components/espressif__esp-tflite-micro/tensorflow/lite/micro/kernels/esp_nn/conv.cc:291 (discriminator 1)
# Disable ESP-NN optimizations for EfficientViT
# Comment out the line: target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)
# in ./managed_components/espressif__esp-tflite-micro/CMakeLists.txt

project(edgebench)