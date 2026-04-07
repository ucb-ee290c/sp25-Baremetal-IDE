/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdbool.h>


#include "riscv.h"
#include "chip_config.h"
#include "rocketcore.h"


/**
 * This section controls which peripheral device is included in the application program.
 * To save the memory space, the unused peripheral device can be commented out.
 */
// #include "hal_core.h"
// #include "hal_clint.h"
// #include "hal_gpio.h"
// #include "hal_i2c.h"
// #include "hal_plic.h"
// #include "hal_uart.h"
//#include "ll_pll.h"

/* USER CODE END Includes */

/* Private defines -----------------------------------------------------------*/
/* USER CODE BEGIN Private defines */

#ifndef BW_NUM_RUNS
#define BW_NUM_RUNS              8u
#endif

#ifndef BW_TARGET_FREQUENCY_HZ
#define BW_TARGET_FREQUENCY_HZ   50000000ULL
#endif

// 0: single-frequency mode
// 1: iterate over BW_PLL_FREQ_LIST frequencies
#ifndef BW_ENABLE_PLL_SWEEP
#define BW_ENABLE_PLL_SWEEP      0
#endif

#ifndef BW_PLL_SWEEP_SLEEP_MS
#define BW_PLL_SWEEP_SLEEP_MS    10000u
#endif

// Comma-separated list used when BW_ENABLE_PLL_SWEEP=1
// #define BW_PLL_FREQ_LIST 50000000ULL, 150000000ULL, 250000000ULL
#ifndef BW_PLL_FREQ_LIST
#define BW_PLL_FREQ_LIST \
  50000000ULL, \
  150000000ULL, \
  250000000ULL, \
  // 350000000ULL 
  // 450000000ULL, \
  // 550000000ULL, \
  // 650000000ULL, \
  // 750000000ULL, \
  // 850000000ULL
#endif

#ifndef BW_BASE_SEED
#define BW_BASE_SEED             0x12345678u
#endif

#ifndef BW_ENABLE_CPU
#define BW_ENABLE_CPU            1
#endif

#ifndef BW_ENABLE_GLIBC
#define BW_ENABLE_GLIBC          1
#endif

#ifndef BW_ENABLE_RVV
#define BW_ENABLE_RVV            1
#endif

#ifndef BW_ENABLE_CPU_MP
#define BW_ENABLE_CPU_MP         1
#endif

#ifndef BW_ENABLE_RVV_MP
#define BW_ENABLE_RVV_MP         1
#endif

#ifndef BW_DRAM_BYTES
#define BW_DRAM_BYTES            (1024u * 1024u)
#endif

#ifndef BW_SCRATCH_BYTES
#define BW_SCRATCH_BYTES         (64u * 1024u)
#endif

#ifndef BW_TCM_BYTES
#define BW_TCM_BYTES             (8u * 1024u)
#endif

#ifndef BW_CACHE_LINE_BYTES
#define BW_CACHE_LINE_BYTES      64u
#endif

#ifndef BW_CACHE_EVICT_BYTES
#define BW_CACHE_EVICT_BYTES     (512u * 1024u)
#endif

#ifndef BW_DRAM_REGION_BASE
#define BW_DRAM_REGION_BASE      0x80000000UL
#endif

#ifndef BW_DRAM_REGION_BYTES
#define BW_DRAM_REGION_BYTES     (256u * 1024u * 1024u)
#endif

#ifndef BW_DRAM_REGION_TOP
#define BW_DRAM_REGION_TOP       (BW_DRAM_REGION_BASE + BW_DRAM_REGION_BYTES)
#endif

/*
 * Place src/dst at the top of DRAM so they:
 * 1) stay far from linked program/heap/stack growth near DRAM base
 * 2) automatically stay non-overlapping as BW_DRAM_BYTES changes.
 */
#ifndef BW_DRAM_DST_BASE
#define BW_DRAM_DST_BASE         (BW_DRAM_REGION_TOP - BW_DRAM_BYTES)
#endif

#ifndef BW_DRAM_SRC_BASE
#define BW_DRAM_SRC_BASE         (BW_DRAM_REGION_TOP - (2u * BW_DRAM_BYTES))
#endif

#if (2u * BW_DRAM_BYTES) > BW_DRAM_REGION_BYTES
#error "BW_DRAM_BYTES too large: need 2*BW_DRAM_BYTES within BW_DRAM_REGION_BYTES"
#endif

#if BW_DRAM_SRC_BASE < BW_DRAM_REGION_BASE
#error "BW_DRAM_SRC_BASE underflows DRAM region"
#endif

#if (BW_DRAM_SRC_BASE + BW_DRAM_BYTES) > BW_DRAM_DST_BASE
#error "DRAM src/dst regions overlap; adjust BW_DRAM_BYTES or DRAM base settings"
#endif

#if (BW_DRAM_DST_BASE + BW_DRAM_BYTES) > BW_DRAM_REGION_TOP
#error "BW_DRAM_DST_BASE+BW_DRAM_BYTES exceeds DRAM region"
#endif

#ifndef BW_SCRATCHPAD_BASE
#define BW_SCRATCHPAD_BASE       0x08000000UL
#endif

#ifndef BW_CORE0_TCM_BASE
#define BW_CORE0_TCM_BASE        0x08010000UL
#endif

#ifndef BW_CORE1_TCM_BASE
#define BW_CORE1_TCM_BASE        0x08012000UL
#endif

void app_init();
void app_main();

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
