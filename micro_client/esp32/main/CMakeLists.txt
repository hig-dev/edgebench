idf_component_register(SRCS 
                        "main.cc"
                        "heap_manager.cc"
                        "edge_bench_client.cc"
                    PRIV_REQUIRES esp_wifi nvs_flash mqtt
                    INCLUDE_DIRS ".")