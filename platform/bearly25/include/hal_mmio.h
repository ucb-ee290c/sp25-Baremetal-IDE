/**
 * \file    hal_mmio.h
 * \author  Jasmine Angle | angle@berkeley.edu
 * \brief   MMIO generic register operation driver.
 * \version 0.1
 * 
 * \copyright Copyright (c) 2025
 * 
 */

#ifndef __HAL_MMIO_H__
#define __HAL_MMIO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void reg_write8(uintptr_t addr, uint8_t data);

uint8_t reg_read8(uintptr_t addr);

void reg_write16(uintptr_t addr, uint16_t data);

uint16_t reg_read16(uintptr_t addr);

void reg_write32(uintptr_t addr, uint32_t data);

uint32_t reg_read32(uintptr_t addr);

void reg_write64(unsigned long addr, uint64_t data);

uint64_t reg_read64(unsigned long addr);


#ifdef __cplusplus
}
#endif

#endif // __HAL_MMIO_H__
