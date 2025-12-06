/**
 * \file    hal_conv.h
 * \author  Louie Labata | louiejoshualabata@berkeley.edu
 * \brief   MMIO generic register operation driver for 1D Conv Engine.
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
#include <stddef.h> // For uintptr_t


// 1D Conv Base Address
#define MMIO_BASE   0x08800000

// Register Offset Definitions
#define CONV_INPUT_ADDR   0x00
#define CONV_OUTPUT_ADDR  0x20
#define CONV_KERNEL_ADDR  0x40
#define CONV_STATUS_ADDR  0x6A
#define CONV_START_ADDR   0x6C
#define CONV_CLEAR_ADDR   0x6D // Set to 1 to flush Datapath

#define CONV_COUNT_ADDR   0x70
#define CONV_LENGTH_ADDR  0x78
#define CONV_DILATION_ADDR 0x7C

#define READ_CHECK_ADDR   0x8D
#define CONV_KERNEL_LEN_ADDR  0x8E
#define CONV_MMIO_RESET   0x8F

// Methods //

void reg_write8(uintptr_t addr, uint8_t data);

uint8_t reg_read8(uintptr_t addr);

void reg_write16(uintptr_t addr, uint16_t data);

uint16_t reg_read16(uintptr_t addr);

void reg_write32(uintptr_t addr, uint32_t data);

uint32_t reg_read32(uintptr_t addr);

void reg_write64(unsigned long addr, uint64_t data);

uint64_t reg_read64(unsigned long addr);

void conv_init();

int conv_set_params(uint32_t* input, uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length);

void conv_read_output(uint32_t *output, int output_len, int *status, uint32_t* input);

void start_conv();

uint8_t get_register_status();

uint32_t get_register_out_count();

uint8_t get_register_read_check();
 

#ifdef __cplusplus
}
#endif

#endif // __HAL_CONV_H__