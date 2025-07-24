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

#define I2C_DELAY 3

static const char *TAG = "I2CComm";

const static uint16_t CRC16_MAXIM_TABLE[256] = {
  0x0000, 0xc0c1, 0xc181, 0x0140, 0xc301, 0x03c0, 0x0280, 0xc241, 0xc601, 0x06c0, 0x0780, 0xc741, 0x0500, 0xc5c1,
  0xc481, 0x0440, 0xcc01, 0x0cc0, 0x0d80, 0xcd41, 0x0f00, 0xcfc1, 0xce81, 0x0e40, 0x0a00, 0xcac1, 0xcb81, 0x0b40,
  0xc901, 0x09c0, 0x0880, 0xc841, 0xd801, 0x18c0, 0x1980, 0xd941, 0x1b00, 0xdbc1, 0xda81, 0x1a40, 0x1e00, 0xdec1,
  0xdf81, 0x1f40, 0xdd01, 0x1dc0, 0x1c80, 0xdc41, 0x1400, 0xd4c1, 0xd581, 0x1540, 0xd701, 0x17c0, 0x1680, 0xd641,
  0xd201, 0x12c0, 0x1380, 0xd341, 0x1100, 0xd1c1, 0xd081, 0x1040, 0xf001, 0x30c0, 0x3180, 0xf141, 0x3300, 0xf3c1,
  0xf281, 0x3240, 0x3600, 0xf6c1, 0xf781, 0x3740, 0xf501, 0x35c0, 0x3480, 0xf441, 0x3c00, 0xfcc1, 0xfd81, 0x3d40,
  0xff01, 0x3fc0, 0x3e80, 0xfe41, 0xfa01, 0x3ac0, 0x3b80, 0xfb41, 0x3900, 0xf9c1, 0xf881, 0x3840, 0x2800, 0xe8c1,
  0xe981, 0x2940, 0xeb01, 0x2bc0, 0x2a80, 0xea41, 0xee01, 0x2ec0, 0x2f80, 0xef41, 0x2d00, 0xedc1, 0xec81, 0x2c40,
  0xe401, 0x24c0, 0x2580, 0xe541, 0x2700, 0xe7c1, 0xe681, 0x2640, 0x2200, 0xe2c1, 0xe381, 0x2340, 0xe101, 0x21c0,
  0x2080, 0xe041, 0xa001, 0x60c0, 0x6180, 0xa141, 0x6300, 0xa3c1, 0xa281, 0x6240, 0x6600, 0xa6c1, 0xa781, 0x6740,
  0xa501, 0x65c0, 0x6480, 0xa441, 0x6c00, 0xacc1, 0xad81, 0x6d40, 0xaf01, 0x6fc0, 0x6e80, 0xae41, 0xaa01, 0x6ac0,
  0x6b80, 0xab41, 0x6900, 0xa9c1, 0xa881, 0x6840, 0x7800, 0xb8c1, 0xb981, 0x7940, 0xbb01, 0x7bc0, 0x7a80, 0xba41,
  0xbe01, 0x7ec0, 0x7f80, 0xbf41, 0x7d00, 0xbdc1, 0xbc81, 0x7c40, 0xb401, 0x74c0, 0x7580, 0xb541, 0x7700, 0xb7c1,
  0xb681, 0x7640, 0x7200, 0xb2c1, 0xb381, 0x7340, 0xb101, 0x71c0, 0x7080, 0xb041, 0x5000, 0x90c1, 0x9181, 0x5140,
  0x9301, 0x53c0, 0x5280, 0x9241, 0x9601, 0x56c0, 0x5780, 0x9741, 0x5500, 0x95c1, 0x9481, 0x5440, 0x9c01, 0x5cc0,
  0x5d80, 0x9d41, 0x5f00, 0x9fc1, 0x9e81, 0x5e40, 0x5a00, 0x9ac1, 0x9b81, 0x5b40, 0x9901, 0x59c0, 0x5880, 0x9841,
  0x8801, 0x48c0, 0x4980, 0x8941, 0x4b00, 0x8bc1, 0x8a81, 0x4a40, 0x4e00, 0x8ec1, 0x8f81, 0x4f40, 0x8d01, 0x4dc0,
  0x4c80, 0x8c41, 0x4400, 0x84c1, 0x8581, 0x4540, 0x8701, 0x47c0, 0x4680, 0x8641, 0x8201, 0x42c0, 0x4380, 0x8341,
  0x4100, 0x81c1, 0x8081, 0x4040};

__attribute__((weak)) uint16_t el_crc16_maxim(const uint8_t* data, size_t length) {
    uint16_t crc = 0x0000;

    for (size_t i = 0; i < length; ++i) {
        uint8_t index = static_cast<uint8_t>(crc ^ data[i]);
        crc           = (crc >> 8) ^ CRC16_MAXIM_TABLE[index];
    }

    return crc ^ 0xffff;
}

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
    packet[2] = data_len >> 8; // High byte of data length
    packet[3] = data_len & 0xFF; // Low byte of data length

    // Copy the data into the packet
    memcpy(packet.data() + header_len, data, data_len);

    // Calculate CRC (not implemented in this example)
    uint16_t crc = el_crc16_maxim(packet.data(), header_len + data_len);
    packet[header_len + data_len] = crc >> 8; // High byte of CRC
    packet[header_len + data_len + 1] = crc & 0xFF; // Low byte of CRC

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
    //ESP_LOGI(TAG, "Writing to I2C device, payload length: %d, total length: %d", data_len, write_buf.size());
    auto ret = i2c_master_transmit(dev_handle_, write_buf.data(), write_buf.size(), -1);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "I2C write failed: %s", esp_err_to_name(ret));
    }

    return ret;
}

esp_err_t I2CComm::write(uint8_t feature, uint8_t cmd, int data_len, uint8_t *data)
{
    vTaskDelay(I2C_DELAY / portTICK_PERIOD_MS);
    if (data_len <= MAX_PL_LEN)
    {
        return write_(feature, cmd, data_len, data);
    }

    auto crc = el_crc16_maxim(data, data_len);
    ESP_LOGI(TAG, "Write big data, crc: %04X", crc);

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
        //ESP_LOGI(TAG, "Writing chunk %d/%d, size: %d, offset: %d", i + 1, total_chunks, chunk_size, offset);
        esp_err_t ret = write_(feature, cmd, chunk_size + 4, chunk_data);
        free(chunk_data);
        if (ret != ESP_OK)
        {
            return ret;
        }
        vTaskDelay(I2C_DELAY / portTICK_PERIOD_MS); // Small delay to avoid flooding the bus
    }
    return ESP_OK;
}

int I2CComm::read_latency_result_ms()
{
    vTaskDelay(I2C_DELAY / portTICK_PERIOD_MS);
    if (!initialized_)
    {
        ESP_LOGE(TAG, "I2C not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t b = 0;
    auto write_ret = write_(I2CCOMM_FEATURE_LATENCY_RESULT, 0, 1, &b);

    vTaskDelay(I2C_DELAY / portTICK_PERIOD_MS); // Small delay to allow processing

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

    uint16_t crc_received = (static_cast<uint16_t>(read_buffer[read_buffer.size() - 2]) << 8) |
                              static_cast<uint16_t>(read_buffer[read_buffer.size() - 1]);  
    uint16_t crc_calculated = el_crc16_maxim(read_buffer.data(), 4+4);  
    if (crc_received != crc_calculated)
    {
        ESP_LOGE(TAG, "Latency result CRC mismatch: received=%04X, calculated=%04X", crc_received, crc_calculated);
        return ESP_ERR_INVALID_CRC;
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
    vTaskDelay(I2C_DELAY / portTICK_PERIOD_MS);
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

    const int max_chunk_size = 192;

    for (int offset = 0; offset < model_output_size; offset += max_chunk_size)
    {
        vTaskDelay(I2C_DELAY / portTICK_PERIOD_MS); // Small delay to avoid flooding the bus
        int chunk_size = std::min(max_chunk_size, model_output_size - offset);
        uint8_t write_buffer[8] =
        {
            uint8_t(offset >> 24),
            uint8_t(offset >> 16),
            uint8_t(offset >> 8),
            uint8_t(offset),
            uint8_t(chunk_size >> 24),
            uint8_t(chunk_size >> 16),
            uint8_t(chunk_size >> 8),
            uint8_t(chunk_size)
        };
        auto transmit_ret = write_(I2CCOMM_FEATURE_ACCURACY_RESULT, 0, 8, write_buffer);
        if (transmit_ret != ESP_OK)
        {
            ESP_LOGE(TAG, "I2C write failed: %s", esp_err_to_name(transmit_ret));
            return std::vector<uint8_t>();
        }

        auto read_buffer = std::vector<uint8_t>(chunk_size + 6); // 6 bytes for header and checksum

        vTaskDelay(I2C_DELAY / portTICK_PERIOD_MS); // Small delay to allow processing
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
        uint16_t crc_received = (static_cast<uint16_t>(read_buffer[read_buffer.size() - 2]) << 8) |
                              static_cast<uint16_t>(read_buffer[read_buffer.size() - 1]);  
        uint16_t crc_calculated = el_crc16_maxim(read_buffer.data(), 4 + chunk_size);               
        if (feature != I2CCOMM_FEATURE_ACCURACY_RESULT || cmd != 0 || data_len != chunk_size)
        {
            ESP_LOGE(TAG, "Invalid accuracy response: feature=%02X, cmd=%02X, data_len=%d", feature, cmd, data_len);
        }
        if (crc_received != crc_calculated)
        {
            ESP_LOGE(TAG, "Accuracy result CRC mismatch: received=%04X, calculated=%04X", crc_received, crc_calculated);
            //return std::vector<uint8_t>();
        }   

        //ESP_LOGI(TAG, "Read accuracy result: offset=%d, size=%d", offset, chunk_size);
        memcpy(output_buffer.data() + offset, read_buffer.data() + 4, chunk_size);
    }
    // Calculate CRC16
    auto crc = el_crc16_maxim(output_buffer.data(), output_buffer.size());
    ESP_LOGI(TAG, "Read accuracy result: size=%zu, CRC=%04X", output_buffer.size(), crc);
    return output_buffer;
}