idf_component_register(SRCS 
                        "main.cc"
                        "edge_bench_client.cc"
                        "../../shared/tensorflow_config.cc"
                    PRIV_REQUIRES esp_wifi nvs_flash mqtt
                    INCLUDE_DIRS
                        "." "../../shared")