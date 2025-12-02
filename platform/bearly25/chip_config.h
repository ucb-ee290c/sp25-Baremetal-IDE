#ifndef __CHIP_CONFIG_H
#define __CHIP_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "riscv.h"
#include "clint.h"
#include "plic.h"
#include "htif.h"
#include "spi.h"
#include "i2c.h"
#include "uart.h"
#include "gpio.h"
#include "pll.h"

// ================================
//  Platform Drivers
// ================================
#include "hal_mmio.h"
#include "hal_rcc.h"
#include "hal_ope.h"


// ================================
//  System Clock
// ================================
// system clock frequency in Hz
#define SYS_CLK_FREQ   100000000

// CLINT time base frequency in Hz
#define MTIME_FREQ     10000


// ================================
//  MMIO devices
// ================================
#define UART_BASE                0x10013000U
#define RCC_BASE                 0x00100000U
#define CLINT_BASE               0x02000000U
#define PLIC_BASE                0x0C000000U
#define TCM_BASE                 0x78000000U
#define SCRATCHPAD_BASE          0x08000000U
#define DRAM_BASE                0x80000000U

#define RCC_CLOCK_SELECTOR       ((ClockSel_Type*)(RCC_BASE + 0x30000))
#define CLINT                    ((CLINT_Type *)CLINT_BASE)
#define PLIC                     ((PLIC_Type *)PLIC_BASE)
#define PLIC_CC                  ((PLIC_ContextControl_Type *)(PLIC_BASE + 0x00200000U))

#define UART0_BASE               (UART_BASE)
#define UART0                    ((UART_Type *)UART0_BASE)

#define CONV2D_BASE              0x08808000U
#define CONV2D                   ((Conv2D_Accel_Type *)CONV2D_BASE)


#ifdef __cplusplus
}
#endif

#endif // __CHIP_CONFIG_H
