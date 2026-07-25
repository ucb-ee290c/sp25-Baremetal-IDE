#ifndef HELLO_WFI_CONFIG_H
#define HELLO_WFI_CONFIG_H

#include <stdio.h>

#include "chip_config.h"

#ifndef HELLO_WFI_LOG_ENABLE
#define HELLO_WFI_LOG_ENABLE 1
#endif

#if HELLO_WFI_LOG_ENABLE
#define HELLO_WFI_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define HELLO_WFI_LOG(...) do { } while (0)
#endif

#ifndef HELLO_WFI_TARGET_FREQUENCY_HZ
#define HELLO_WFI_TARGET_FREQUENCY_HZ 500000000ULL
#endif

/* How many "hello world" lines to print before parking in the wfi loop. */
#ifndef HELLO_WFI_HELLO_COUNT
#define HELLO_WFI_HELLO_COUNT 10u
#endif

/* Machine software-interrupt pending register (MSIP) for hart 0 lives at CLINT offset 0.
 * Manually write a nonzero word here (e.g. from OpenOCD / the host) to wake the wfi loop;
 * the app clears it on wake. Same register the bootrom uses to release harts. */
#ifndef HELLO_WFI_MSIP_ADDR
#define HELLO_WFI_MSIP_ADDR (CLINT_BASE + 0x0000U)
#endif

#endif /* HELLO_WFI_CONFIG_H */
