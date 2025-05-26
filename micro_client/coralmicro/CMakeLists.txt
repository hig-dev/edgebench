cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake)
endif()

project(coralmicro-app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(coralmicro)
add_subdirectory(coralmicro)

# Include coreMQTT paths
include(coreMQTT/mqttFilePaths.cmake)

# Include coreMQTT headers
include_directories(${MQTT_INCLUDE_PUBLIC_DIRS})

add_executable_m7(coralmicro-app
        main.cc
        mqtt.cc
        edge_bench_client.cc
        ${MQTT_SOURCES}
        ${MQTT_SERIALIZER_SOURCES}
)

target_compile_definitions(coralmicro-app
    PRIVATE
      MQTT_DO_NOT_USE_CUSTOM_CONFIG
)

target_link_libraries(coralmicro-app
        libs_base-m7_freertos
        libs_base-m7_wifi
)
