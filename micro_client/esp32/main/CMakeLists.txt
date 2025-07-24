idf_component_register(SRCS 
                        "main.cc"
                        "edge_bench_client.cc"
                        "i2c_comm.cc"
                        "../../shared/tensorflow_config.cc"
                    PRIV_REQUIRES esp_wifi nvs_flash mqtt esp_driver_i2c esp_driver_gpio
                    INCLUDE_DIRS
                        "." "../../shared")