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

#include <stdint.h>

// Baremetal IDE Definitions //
#include "metal.h"

#define MMIO_BASE        0x08800000

// Methods //

void reg_write8(uintptr_t addr, uint8_t data);

uint8_t reg_read8(uintptr_t addr);

void reg_write16(uintptr_t addr, uint16_t data);

uint16_t reg_read16(uintptr_t addr);

void reg_write32(uintptr_t addr, uint32_t data);

uint32_t reg_read32(uintptr_t addr);

void reg_write64(unsigned long addr, uint64_t data);

uint64_t reg_read64(unsigned long addr);

/**
 * \brief Simplified wrapper function for performing 2D convolution
 * \param inputAddr Source address of the input data
 * \param outputAddr Destination address for the output data
 * \param kernelAddr Pointer to the kernel values
 * \param lengthAddr Pointer to the number of entries in the input data 
 * \param dilationAddr ___(TODO: FILL OUT DOCUMENTATION)___
 * This Function returns a uint8_t, which is a status flag (i.e. error flags)
 * 
 * Here's an example of how to call it:
 * 
  uint8_t status = perform_convolution((uint64_t)&testImage, (uint64_t)&outputImage, HEIGHT, WIDTH, (uint8_t*)kernel, 3, (uint8_t)0, 1);
  
  if (status != 0x0) {
    printf("Error Status: %p\n", status);
  }
 * 
 * \
 * If the status is 0x0, That means it was successful.
 */
 
uint8_t perform_convolution(uint64_t inputAddrValue,     
                            uint64_t outputAddrValue, 
                            uint16_t kernelAddrValue, 
                            uint32_t lengthAddrValue, 
                            uint16_t dilationAddrValue);


/*

TODO: figure out the rest of these things!
Not sure about using the rest in any other functions?

 typedef struct {
   __IO uint32_t OUT_COUNT;     // 0x70: Number of output elements ready
   __I  uint8_t  READ_CHECK;    // 0x8D: DMA Read check status
   __IO uint8_t  KERNEL_LEN;    // 0x8E: Kernel length 16 (1) or 8 (0)?
   __IO uint8_t  MMIO_RESET;    // 0x8F: Reset MMIO 
 } ConvAccel_Type;
*/


#ifdef __cplusplus
}
#endif

#endif // __HAL_CONV_H__