#ifndef MEMLAT_ADDRS_H
#define MEMLAT_ADDRS_H

#include <stdint.h>
#include "chip_config.h"

#ifndef SCRATCHPAD_SIZE
#define SCRATCHPAD_SIZE (64UL * 1024UL)
#endif

#ifndef SCRATCHPAD_BASE
#define SCRATCHPAD_BASE (0x80000000UL)
#endif

#ifndef TCM_SIZE
#define TCM_SIZE (8UL * 1024UL)
#endif

#define DRAM_PTR(base_offset_bytes) \
    ((volatile uint32_t *)((uintptr_t)DRAM_BASE + (uintptr_t)(base_offset_bytes)))

#define SCRATCHPAD_PTR(base_offset_bytes) \
    ((volatile uint32_t *)((uintptr_t)SCRATCHPAD_BASE + (uintptr_t)(base_offset_bytes)))

#define TCM_PTR(base_offset_bytes) \
    ((volatile uint32_t *)((uintptr_t)TCM_BASE + (uintptr_t)(base_offset_bytes)))

#endif // MEMLAT_ADDRS_H
