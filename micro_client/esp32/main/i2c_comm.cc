#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c_master.h"
#include "i2c_comm.h"
#include <algorithm>

static const char *TAG = "I2CComm";

I2CComm::I2CComm(uint8_t i2c_address)
    : i2c_address_(i2c_address), bus_handle_(nullptr), dev_handle_(nullptr), initialized_(false)
{
}

I2CComm::~I2CComm()
{
    if (initialized_)
    {
        // Clean up I2C resources
        if (dev_handle_)
        {
            i2c_master_bus_rm_device(dev_handle_);
        }
        if (bus_handle_)
        {
            i2c_del_master_bus(bus_handle_);
        }
    }
}

esp_err_t I2CComm::init()
{
    if (initialized_)
    {
        ESP_LOGW(TAG, "I2C already initialized");
        return ESP_OK;
    }

    i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_MASTER_NUM,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .intr_priority = 0,
        .trans_queue_depth = 0,
        .flags = {
            .enable_internal_pullup = true,
            .allow_pd = false,
        },
    };

    esp_err_t ret = i2c_new_master_bus(&bus_config, &bus_handle_);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to create I2C bus: %s", esp_err_to_name(ret));
        return ret;
    }

    i2c_device_config_t dev_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = i2c_address_,
        .scl_speed_hz = I2C_MASTER_FREQ_HZ,
        .scl_wait_us = 12 * 1000,
        .flags = {
            .disable_ack_check = false,
        },
    };

    ret = i2c_master_bus_add_device(bus_handle_, &dev_config, &dev_handle_);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to add device to I2C bus: %s", esp_err_to_name(ret));
        i2c_del_master_bus(bus_handle_);
        bus_handle_ = nullptr;
        return ret;
    }

    initialized_ = true;
    ESP_LOGI(TAG, "I2C initialized successfully");
    return ESP_OK;
}

std::vector<uint8_t> I2CComm::create_packet_(uint8_t feature, uint8_t cmd, uint16_t data_len, uint8_t *data)
{
    if (data_len > MAX_PL_LEN)
    {
        ESP_LOGE(TAG, "Data length exceeds maximum allowed size");
        return std::vector<uint8_t>();
    }

    // Allocate memory for the packet
    const int header_len = 4; // 1 byte for feature, 1 byte for command, 2 bytes for data length
    const int crc_len = 2;    // CRC length
    int total_len = header_len + data_len + crc_len;
    auto packet = std::vector<uint8_t>(total_len);

    // Fill the packet header
    packet[0] = feature; // Feature byte
    packet[1] = cmd;     // Command byte
    packet[2] = data_len >> 8;
    packet[3] = data_len & 0xFF; // Low byte of data length

    // Copy the data into the packet
    memcpy(packet.data() + header_len, data, data_len);

    // Calculate CRC (not implemented in this example)
    packet[header_len + data_len] = 0xFF;
    packet[header_len + data_len + 1] = 0xFF;

    return packet;
}

esp_err_t I2CComm::write_(uint8_t feature, uint8_t cmd, uint16_t data_len, uint8_t *data)
{
    if (!initialized_)
    {
        ESP_LOGE(TAG, "I2C not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    std::vector<uint8_t> write_buf = create_packet_(feature, cmd, data_len, data);
    ESP_LOGI(TAG, "Writing to I2C device, payload length: %d, total length: %d", data_len, write_buf.size());
    auto ret = i2c_master_transmit(dev_handle_, write_buf.data(), write_buf.size(), -1);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C write failed: %s", esp_err_to_name(ret));
    }

    return ret;
}

esp_err_t I2CComm::write(uint8_t feature, uint8_t cmd, int data_len, uint8_t *data)
{
    if (data_len <= MAX_PL_LEN)
    {
        return write_(feature, cmd, data_len, data);
    }

    // Send data in chunks if it exceeds the maximum length
    // Chunk format: offset (4 bytes), data
    int effective_max_pl_len = MAX_PL_LEN - 4; // 4 bytes for offset
    int total_chunks = (data_len + effective_max_pl_len - 1) / effective_max_pl_len;
    ESP_LOGI(TAG, "Data length %d exceeds max payload length %d, splitting into %d chunks", data_len, MAX_PL_LEN, total_chunks);
    for (int i = 0; i < total_chunks; ++i)
    {
        int offset = i * effective_max_pl_len;
        int chunk_size = std::min(effective_max_pl_len, data_len - offset);

        // Prepare the chunk data
        uint8_t *chunk_data = (uint8_t *)malloc(chunk_size + 4);
        if (chunk_data == NULL)
        {
            ESP_LOGE(TAG, "Failed to allocate memory for chunk data");
            return ESP_ERR_NO_MEM;
        }

        // Set the offset in the first 4 bytes
        chunk_data[0] = uint8_t(offset >> 24);
        chunk_data[1] = uint8_t(offset >> 16);
        chunk_data[2] = uint8_t(offset >> 8);
        chunk_data[3] = uint8_t(offset);

        // Copy the actual data
        memcpy(chunk_data + 4, data + offset, chunk_size);
        ESP_LOGI(TAG, "Writing chunk %d/%d, size: %d, offset: %d",
                 i + 1, total_chunks, chunk_size, offset);
        esp_err_t ret = write_(feature, cmd, chunk_size + 4, chunk_data);
        free(chunk_data);
        if (ret != ESP_OK)
        {
            return ret;
        }
        vTaskDelay(5 / portTICK_PERIOD_MS); // Small delay to avoid flooding the bus
    }
    return ESP_OK;
}

int I2CComm::read_latency_result_ms()
{
    if (!initialized_)
    {
        ESP_LOGE(TAG, "I2C not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t b = 0;
    auto write_ret = write_(I2CCOMM_FEATURE_LATENCY_RESULT, 0, 1, &b);

    vTaskDelay(10 / portTICK_PERIOD_MS); // Small delay to allow processing

    auto read_buffer = std::vector<uint8_t>(10); // 1 byte feature, 1 byte cmd, 2 bytes data_len, 4 bytes latency, 2 bytes crc
    auto read_ret = i2c_master_receive(dev_handle_, read_buffer.data(), read_buffer.size(), -1);
    if (read_ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C read failed: %s", esp_err_to_name(read_ret));
        return read_ret;
    }

    uint8_t feature = read_buffer[0];
    uint8_t cmd = read_buffer[1];
    uint16_t data_len = (static_cast<uint16_t>(read_buffer[2]) << 8) |
                        static_cast<uint16_t>(read_buffer[3]);
    if (feature != I2CCOMM_FEATURE_LATENCY_RESULT || cmd != 0 || data_len != 4)
    {
        ESP_LOGE(TAG, "Invalid latency response: feature=%02X, cmd=%02X, data_len=%d", feature, cmd, data_len);
    }

    int ms = (static_cast<int>(read_buffer[4]) << 24) |
             (static_cast<int>(read_buffer[5]) << 16) |
             (static_cast<int>(read_buffer[6]) << 8) |
             static_cast<int>(read_buffer[7]);

    ESP_LOGI(TAG, "Read latency result: %d ms", ms);

    return ms;
}

std::vector<uint8_t> I2CComm::read_accuracy_result(int model_output_size)
{
    if (!initialized_)
    {
        ESP_LOGE(TAG, "I2C not initialized");
        return std::vector<uint8_t>();
    }

    auto output_buffer = std::vector<uint8_t>(model_output_size);
    if (output_buffer.size() != DEFAULT_OUTPUT_SIZE)
    {
        ESP_LOGE(TAG, "Output buffer size mismatch: expected %d, got %zu", DEFAULT_OUTPUT_SIZE, output_buffer.size());
        return std::vector<uint8_t>();
    }

    for (int offset = 0; offset < model_output_size; offset += MAX_PL_LEN)
    {
        vTaskDelay(10 / portTICK_PERIOD_MS); // Small delay to avoid flooding the bus
        uint8_t offset_buffer[4] = {
            uint8_t(offset >> 24),
            uint8_t(offset >> 16),
            uint8_t(offset >> 8),
            uint8_t(offset)};
        auto transmit_ret = write_(I2CCOMM_FEATURE_ACCURACY_RESULT, 0, 4, offset_buffer);
        if (transmit_ret != ESP_OK)
        {
            ESP_LOGE(TAG, "I2C write failed: %s", esp_err_to_name(transmit_ret));
            return std::vector<uint8_t>();
        }

        int chunk_size = std::min(MAX_PL_LEN, model_output_size - offset);
        auto read_buffer = std::vector<uint8_t>(chunk_size + 6); // 6 bytes for header and checksum

        vTaskDelay(10 / portTICK_PERIOD_MS); // Small delay to allow processing
        auto read_ret = i2c_master_receive(dev_handle_, read_buffer.data(), read_buffer.size(), -1);
        if (read_ret != ESP_OK)
        {
            ESP_LOGE(TAG, "I2C read failed: %s", esp_err_to_name(read_ret));
            return std::vector<uint8_t>();
        }

        uint8_t feature = read_buffer[0];
        uint8_t cmd = read_buffer[1];
        uint16_t data_len = (static_cast<uint16_t>(read_buffer[2]) << 8) |
                            static_cast<uint16_t>(read_buffer[3]);
        if (feature != I2CCOMM_FEATURE_ACCURACY_RESULT || cmd != 0 || data_len != chunk_size)
        {
            ESP_LOGE(TAG, "Invalid accuracy response: feature=%02X, cmd=%02X, data_len=%d", feature, cmd, data_len);
        }

        ESP_LOGI(TAG, "Read accuracy result: offset=%d, size=%d", offset, chunk_size);
        memcpy(output_buffer.data() + offset, read_buffer.data() + 4, chunk_size);
    }
    return output_buffer;
}