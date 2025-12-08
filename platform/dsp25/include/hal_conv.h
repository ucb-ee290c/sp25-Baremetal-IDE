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
// #include <stdio.h>
// #include <stdint.h>
#include <string.h>

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

// Convolution Status Struct

// Define the bit masks using an enum for clarity and maintainability
typedef enum {
    STATUS_BUSY      = 1 << 0,  // Bit 0 (0x01)
    STATUS_COMPL     = 1 << 1,  // Bit 1 (0x02)
    STATUS_ERROR     = 1 << 2,  // Bit 2 (0x04)
    STATUS_INVALID   = 1 << 3,  // Bit 3 (0x08)
    STATUS_INFINITE  = 1 << 4,  // Bit 4 (0x10)
    STATUS_OVERFLOW  = 1 << 5,  // Bit 5 (0x20)
    STATUS_UNDERFLOW = 1 << 6,  // Bit 6 (0x40)
    STATUS_INEXACT   = 1 << 7   // Bit 7 (0x80)
} ConvStatusBits;

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

uint8_t conv_read_output(uint32_t *output, uint32_t output_len);

void start_conv();

uint8_t get_register_status();

uint32_t get_register_out_count();

uint8_t get_register_read_check();


/**
 * \brief Simplified wrapper function for performing 2D convolution
 * \param srcAddrValue Source address of the input data
 * \param destAddrValue Destination address for the output data
 * \param inputHeightValue Input height
 * \param inputWidthValue Input width
 * \param kernel Pointer to the kernel values
 * \param kernelSizeValue Size of the kernel (must be 3 or 5)
 * \param useReLU Whether to use ReLU activation
 * \param strideValue Stride value for the convolution
 * This Function returns a uint8_t, which is a status flag (i.e. error flags)
 * 
 * Here's an example of how to call it:
 * 
  uint8_t status = perform_convolution((uint64_t)&testImage, (uint64_t)&outputImage, HEIGHT, WIDTH, (uint8_t*)kernel, 3, (uint8_t)0, 1);
  
  if (status != 0x0) {
    printf("Error Status: %p\n", status);
  }
 * 
 * If the status is 0x0, That means it was successful.
 int: 0 on success, -1 if parameters are invalid (e.g., unsupported kernel length).
 */

uint8_t perform_convolution_1D(uint32_t* input, uint32_t input_length, uint32_t* kernel, uint8_t kernel_length, uint32_t* output, uint16_t dilation);

// TODO: turn the below in to \param and \brief 
/* 
Computes the convolution of arr with the given kernel and dilation factor and stores the result in output, specifically 
based on the implementation of the convolution block. The first value in the output array is computed with the kernel's 
left element aligned with the array's left element.

arr:        pointer to input array      FP16 array
arr_len:    length of input array       
kernel:     pointer to kernel array     FP16 array (represented as uint16_t)
kernel_len: length of kernel array 
dilation:   dilation factor
output:     pointer to output array     FP16 array (represented as uint16_t) 

Example input and output: 

arr:        {1, 2, 3, 4}
arr_len:    4
kernel:     {-1, 1, -1}
kernel_len: 3
dilation:   1

output: {-2, -3, -1, -4} ({-1*1 + 1*2 + -1*3, -1*2 + 1*3 + -1*4, -1*3 + 1*4 + -1*0, -1*4 + 1*0 + -1*0})

For border values (at the end), we assume the array is zero-extended to fit the length of the kernel (including dilation).
*/
                   
void perform_naive_convolution_1D(uint32_t *arr, size_t arr_len, uint32_t *kernel, size_t kernel_len, size_t dilation, float *output);

// Returns a human-readable string containing all set status flags
void get_register_status_human_readable(char* buffer);

#ifdef __cplusplus
}
#endif

#endif // __HAL_CONV_H__