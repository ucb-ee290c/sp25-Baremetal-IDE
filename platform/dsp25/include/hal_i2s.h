#ifndef HAL_I2S_H
#define HAL_I2S_H

#ifdef __cplusplus
extern "C" {
#endif

#include "chip_config.h"

// ================================
//  MMIO Address Definitions
// ================================

#define I2S_CONFIG(channel)                  (I2S_BASE + 0x2 * channel)
#define I2S_STATUS(channel)                  (I2S_BASE + 0x08 + channel)

// Watermark signal values (read only)
#define I2S_WATERMARK_RX_L(channel)          (I2S_BASE + 0x102 + 0x4 * channel)
#define I2S_WATERMARK_RX_R(channel)          (I2S_BASE + 0x103 + 0x4 * channel)
#define I2S_WATERMARK_TX_L(channel)          (I2S_BASE + 0x100 + 0x4 * channel)
#define I2S_WATERMARK_TX_R(channel)          (I2S_BASE + 0x101 + 0x4 * channel)

// Set watermark level (read/write)
#define I2S_TX_WATERMARK(channel)            (I2S_BASE + 0x0C + channel)
#define I2S_RX_WATERMARK(channel)            (I2S_BASE + 0x10 + channel)

#define I2S_STATUS(channel)                  (I2S_BASE + 0x08 + channel)

#define I2S_CLKDIV(channel)                  (I2S_BASE + 0x14 + 0x2 * channel)

#define I2S_TX_L(channel)                    (I2S_BASE + 0x20 + 0x10 * channel)
#define I2S_TX_R(channel)                    (I2S_BASE + 0x28 + 0x10 * channel)
#define I2S_RX_L(channel)                    (I2S_BASE + 0x60 + 0x10 * channel)
#define I2S_RX_R(channel)                    (I2S_BASE + 0x68 + 0x10 * channel)

typedef enum {
    I2S_LEFT = 0,
    I2S_RIGHT = 1
} i2s_channel_side_t;

typedef struct i2s_params {
    int tx_en;                             // Enable I2S TX enable
    int rx_en;                             // Enable I2S RX enable
    int bitdepth_tx;                       // Bitdepth for TX (0:8bit, 1:16bit, 2:24bit, 3:32bit)
    int bitdepth_rx;                       // Bitdepth for RX (0:8bit, 1:16bit, 2:24bit, 3:32bit)
    int ws_len;                            // Word select length (0:16, 1:24, 2:32, 3:48). Should match bitdepth
    int clkdiv;                            // Clock divider for generating I2S clock from system clock
    int clkgen;                            // Enable internal clock generator
    int dacen;                             // Enable I2S DAC output
    int tx_fp;                             // Enable TX floating point
    int rx_fp;                             // Enable RX floating point
    int tx_force_left;                     // Force TX to left channel
    int rx_force_left;                     // Force RX to left channel
} i2s_params_t;



// ================================
//  Bitfield Structs
// ================================

typedef struct i2s_config {
    uint16_t clkgen_en : 1;
    uint16_t tx_en : 1;
    uint16_t rx_en : 1;
    uint16_t dac_en : 1;
    uint16_t ws_len : 2;
    uint16_t tx_bitdepth : 2;
    uint16_t rx_bitdepth : 2;
    uint16_t tx_fp_en : 1;
    uint16_t rx_fp_en : 1;
    uint16_t tx_force_left : 1;
    uint16_t rx_force_left : 1;
    uint16_t unused : 2;
} __attribute__((packed)) i2s_config_t;

typedef struct i2s_status {
    uint8_t l_tx_full : 1;
    uint8_t r_tx_full : 1;
    uint8_t l_rx_empty : 1;
    uint8_t r_rx_empty : 1;
    uint8_t unused : 4;
} __attribute__((packed)) i2s_status_t;



// ================================
//  Function Prototypes
// ================================

/* Set I2S parameters based on i2s_params_t structure */
void config_I2S(int channel, i2s_params_t* params);

void set_I2S_clkdiv(int channel, uint16_t clkdiv);

void set_I2S_params(i2s_params_t* params);

void set_I2S_watermark(int channel, int watermark_tx, int watermark_rx);

int get_I2S_rx_watermark(int channel, i2s_channel_side_t left_right);

int get_I2S_tx_watermark(int channel, i2s_channel_side_t left_right);

int get_I2S_tx_full(int channel, i2s_channel_side_t left_right);

// TODO: Unsure if this will be true if there is only 1 sample ready
//      because rx returns 2 samples packed in one 64 bit read
int get_I2S_rx_empty(int channel, i2s_channel_side_t left_right);

void set_I2S_en(int channel, int tx_en, int rx_en);

// TODO: Unsure how samples are packed
uint64_t read_I2S_rx(int channel, i2s_channel_side_t left);

// TODO: Unsure how samples are packed
void write_I2S_tx(int channel, i2s_channel_side_t left, uint64_t data);

// DMA functions - WIP
// uint64_t write_I2S_tx_DMA(int channel, int dma_num, int length, uint64_t* read_addr, int left, int poll);
// uint64_t read_I2S_rx_DMA(int channel, int dma_num, int length, uint64_t* write_addr, int left, int poll);

void set_I2S_fp(int channel, int tx_fp, int rx_fp);

void set_I2S_force_left(int channel, int tx_force_left, int rx_force_left);


#ifdef __cplusplus
}
#endif

#endif // HAL_I2S_H