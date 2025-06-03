# For more information about build system see
# https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html
# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

#set(COMPONENTS main) Uncommented to use PSRAM

add_compile_definitions(
  ESP32=1
  I2C_MASTER=1
  MOBILEONE=1
  EFFICIENTVIT=0
  DEIT=0
)

project(edgebench)