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
 #define CONV_COUNT_ADDR         0x70
 #define CONV_LENGTH_ADDR        0x78
 #define CONV_DILATION_ADDR      0x7C
 #define READ_CHECK_ADDR         0x8D
 #define CONV_KERNEL_LEN_ADDR    0x8E
 #define CONV_MMIO_RESET         0x8F
 
 typedef struct {
   __IO uint64_t INPUT;         // 0x00: Input address/config
   __IO uint64_t OUTPUT;        // 0x20: Output address/config
   __IO uint64_t KERNEL;        // 0x40: Kernel address/config
   __I  uint8_t  STATUS;        // 0x6A: Status register
   __IO uint8_t  START;         // 0x6C: Start command
   __IO uint32_t OUT_COUNT;     // 0x70: Number of output elements ready
   __IO uint32_t LENGTH;        // 0x78: Input length
   __IO uint16_t DILATION;      // 0x7C: Dilation factor
   __I  uint8_t  READ_CHECK;    // 0x8D: DMA Read check status
   __IO uint8_t  KERNEL_LEN;    // 0x8E: Kernel length 16 (1) or 8 (0)?
   __IO uint8_t  MMIO_RESET;    // 0x8F: Reset MMIO 
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
 
 void conv_read_output(ConvAccel_Type *conv, uint32_t *output, int output_len, int status, uint32_t* input);
 
 void start_conv(ConvAccel_Type *conv);
 
 uint8_t get_register_status(ConvAccel_Type *conv);
 
 uint32_t get_register_out_count(ConvAccel_Type *conv);
 
 uint8_t get_register_read_check(ConvAccel_Type *conv);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // __HAL_CONV_H__