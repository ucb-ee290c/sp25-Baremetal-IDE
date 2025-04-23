/**
 * \file    hal_rcc.h
 * \author  Jasmine Angle | angle@berkeley.edu
 * \brief   PRCI RCC driver.
 * \version 0.1
 * 
 * \copyright Copyright (c) 2025
 * 
 */

#ifndef __HAL_RCC_H__
#define __HAL_RCC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "metal.h"

/**
 * \brief   Clock selection mux options.
 * \note    Creates an enum named `ClockSel_Opts` for later use.
 */
typedef enum {
  CLKSEL_SLOW = (uint32_t)0,
  CLKSEL_PLL0 = (uint32_t)1,
  CLKSEL_PLL1 = (uint32_t)2
} ClockSel_Opts;

/**
 * \brief   Describes the memory map of the PRCI clock selector MMIO.
 * \todo    This needs to be confirmed with the RTL.
 */
typedef struct {
  __IO uint32_t UNCORE;                                 // 0x00
  __IO uint32_t TILE0;                                  // 0x04
  __IO uint32_t TILE1;                                  // 0x08
  __IO uint32_t CLKTAP;                                 // 0x0C
} ClockSel_Type;

void set_all_clocks(ClockSel_Type* clksel, ClockSel_Opts clksrc);

#ifdef __cplusplus
}
#endif

#endif // __HAL_RCC_H__
