/**
 * \file    hal_conv.h
 * \brief   1d conv driver.
 * \version 0.1
 * 
 * \copyright Copyright (c) 2025
 * 
 */

 #ifndef __HAL_CONV_H__
 #define __HAL_CONV_H__
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 // Baremetal IDE Definitions //
 #include "metal.h"
 
 #include <stdint.h>
 
 // Register Offset Definitions
 #define CONV_INPUT_ADDR         0x00
 #define CONV_OUTPUT_ADDR        0x20
 #define CONV_KERNEL_ADDR        0x40
 #define CONV_STATUS_ADDR        0x6A
 #define CONV_START_ADDR         0x6C
 #define CONV_CLEAR_ADDR         0x6D

//  Clear 0x6D 1 W Set to 1 to flush Datapath

 #define CONV_COUNT_ADDR         0x70
 #define CONV_LENGTH_ADDR        0x78
 #define CONV_DILATION_ADDR      0x7C

// Req Enqueue 0x8C 1 R Set to 1 by accelerator logic when the input queue is low on data (DMA only)

// Deq Output Valid 0x8D 1 R DMA polls this bit to check if data in the output queue is valid
 #define READ_CHECK_ADDR         0x8D
 #define CONV_KERNEL_LEN_ADDR    0x8E
 #define CONV_MMIO_RESET         0x8F
 
// typedef struct {
//     __IO uint64_t INPUT;       // 0x00
//     uint8_t pad0[0x20 - 0x08];  // padding to 0x20
//     __IO uint64_t OUTPUT;      // 0x20
//     uint8_t pad1[0x40 - 0x28];  // padding to 0x40
//     __IO uint64_t KERNEL;      // 0x40
//     uint8_t pad2[0x6A - 0x48]; // padding to 0x6A
//     __I  uint8_t STATUS;       // 0x6A
//     uint8_t pad3[0x6C - 0x6B]; // padding to 0x6C
//     __IO uint8_t START;        // 0x6C
//     uint8_t pad4[0x70 - 0x6D]; // padding to 0x70
//     __IO uint32_t OUT_COUNT;   // 0x70
//     uint8_t pad5[0x78 - 0x74]; // padding to 0x78
//     __IO uint32_t LENGTH;      // 0x78
//     __IO uint16_t DILATION;    // 0x7C
//     uint8_t pad6[0x8D - 0x7E]; // padding to 0x8D
//     __I  uint8_t READ_CHECK;   // 0x8D
//     __IO uint8_t KERNEL_LEN;   // 0x8E
//     __IO uint8_t MMIO_RESET;   // 0x8F
// } ConvAccel_Type;

typedef struct __attribute__((packed)) {
    __IO uint64_t INPUT;           // 0x00
    uint8_t _pad0[0x20 - 0x08];   // pad until 0x20

    __IO uint64_t OUTPUT;          // 0x20
    uint8_t _pad1[0x40 - 0x28];   // pad until 0x40

    __IO uint64_t KERNEL;          // 0x40
    uint8_t _pad2[0x6A - 0x48];   // pad until 0x6A

    __I  uint8_t  STATUS;          // 0x6A
    uint8_t _pad3[0x6C - 0x6B];   // pad until 0x6C

    __IO uint8_t  START;           // 0x6C
    __IO uint8_t  CLEAR;           // 0x6D
    uint8_t _pad4[0x70 - 0x6E];   // pad until 0x70

    __IO uint32_t OUT_COUNT;       // 0x70
    uint8_t _pad5[0x78 - 0x74];   // pad until 0x78

    __IO uint32_t LENGTH;          // 0x78
    __IO uint16_t DILATION;        // 0x7C
    uint8_t _pad6[0x8C - 0x7E];   // pad until 0x8C

    __I  uint8_t  REQ_ENQ;       // 0x8C
    __I  uint8_t  READ_CHECK;      // 0x8D
    __IO uint8_t  KERNEL_LEN;      // 0x8E
    __IO uint8_t  MMIO_RESET;      // 0x8F
} ConvAccel_Type;
 
 void reg_write8(uintptr_t addr, uint8_t data);
 uint8_t reg_read8(uintptr_t addr);
 void reg_write16(uintptr_t addr, uint16_t data);
 uint16_t reg_read16(uintptr_t addr);
 void reg_write32(uintptr_t addr, uint32_t data);
 uint32_t reg_read32(uintptr_t addr);
 void reg_write64(unsigned long addr, uint64_t data);
 uint64_t reg_read64(unsigned long addr);
 
 void conv_init(ConvAccel_Type *conv);
 
 int conv_set_params(ConvAccel_Type *conv, uint32_t* input, uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length);
 
 void conv_read_output(ConvAccel_Type *conv, uint32_t *output, int output_len, int *status, uint32_t* input);
 
 void start_conv(ConvAccel_Type *conv);
 
 uint8_t get_register_status(ConvAccel_Type *conv);
 
 uint32_t get_register_out_count(ConvAccel_Type *conv);
 
 uint8_t get_register_read_check(ConvAccel_Type *conv);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // __HAL_CONV_H__