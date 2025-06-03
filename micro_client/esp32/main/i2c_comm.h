#ifndef I2C_COMM_H
#define I2C_COMM_H

#include "driver/i2c_master.h"
#include <cstddef>
#include <cstdint>
#include <vector>

#define DEFAULT_I2C_ADDRESS 0x62
#define MAX_PL_LEN 256
#define DEFAULT_OUTPUT_SIZE 65536

/**
 * \enum I2CCOMM_FEATURE_E
 * \brief this enumeration use in i2c communication library, define the supported feature.
 */
typedef enum
{
    I2CCOMM_FEATURE_MODE                 = 0x00, /**< Set test mode*/
    I2CCOMM_FEATURE_ITERATIONS           = 0x01, /**< Set iterations for test mode*/
    I2CCOMM_FEATURE_MODEL                = 0x02, /**< Set model for test mode*/
    I2CCOMM_FEATURE_INPUT                = 0x03, /**< Set input for test mode*/
    I2CCOMM_FEATURE_STATUS               = 0x04, /**< Status*/
    I2CCOMM_FEATURE_CMD                  = 0x05, /**< Command*/
    I2CCOMM_FEATURE_LATENCY_RESULT       = 0x06, /**< Latency result*/
    I2CCOMM_FEATURE_ACCURACY_RESULT      = 0x07, /**< Accuracy result*/
} I2CCOMM_FEATURE_E;


/**
 * @brief C++ class for I2C communication
 */
class I2CComm {
public:
    /**
     * @brief Constructor for I2CComm
     * 
     * @param i2c_address I2C device address, default is 0x62
     */
    I2CComm(uint8_t i2c_address = DEFAULT_I2C_ADDRESS);
    ~I2CComm();

    /**
     * @brief Initialize the I2C master
     * 
     * @return esp_err_t ESP_OK on success
     */
    esp_err_t init();

    /**
     * @brief Write data to an I2C device register
     *
     * @param data Pointer to data buffer
     * @param data_len Length of data to write
     * @return esp_err_t ESP_OK on success
     */
    esp_err_t write(uint8_t feature, uint8_t cmd, int data_len, uint8_t *data);

    int read_latency_result_ms();

    std::vector<uint8_t> read_accuracy_result(int model_output_size);

private:
    // I2C master configuration
    static constexpr gpio_num_t I2C_MASTER_SCL_IO = gpio_num_t::GPIO_NUM_4;
    static constexpr gpio_num_t I2C_MASTER_SDA_IO = gpio_num_t::GPIO_NUM_5;
    static constexpr int I2C_MASTER_NUM = I2C_NUM_0;
    static constexpr uint32_t I2C_MASTER_FREQ_HZ = 400000;
    static constexpr int I2C_MASTER_TX_BUF_DISABLE = 0;
    static constexpr int I2C_MASTER_RX_BUF_DISABLE = 0;
    static constexpr int I2C_MASTER_TIMEOUT_MS = 1000;

    uint8_t i2c_address_;
    i2c_master_bus_handle_t bus_handle_;
    i2c_master_dev_handle_t dev_handle_;
    bool initialized_;

    /**
     * @brief Create a packet for I2C communication
     *
     * @param feature Feature type
     * @param cmd Command type
     * @param data_len Length of data
     * @param data Pointer to data buffer
     * @return uint8_t* Pointer to the created packet
     */
    std::vector<uint8_t> create_packet_(uint8_t feature, uint8_t cmd, uint16_t data_len, uint8_t *data);

    /**
     * @brief Write data to an I2C device register
     *
     * @param data Pointer to data buffer
     * @param data_len Length of data to write
     * @return esp_err_t ESP_OK on success
     */
    esp_err_t write_(uint8_t feature, uint8_t cmd, uint16_t data_len, uint8_t *data);
};

#endif /* I2C_COMM_H */
