/*
 * memlat_addrs.h - Memory region addresses for Bearly25.
 *
 * From the generated DTS / memmap:
 *   memory@8000000   -> scratchpad  64 KB on MBUS
 *   memory@8010000   -> Core 0 TCM   8 KB  (2 banks)
 *   memory@8012000   -> Core 1 TCM   8 KB  (2 banks)
 *   memory@80000000  -> DRAM
*/

#ifndef MEMLAT_ADDRS_H
#define MEMLAT_ADDRS_H

#include <stdint.h>

#define DRAM_BASE               0x80000000UL

#define SCRATCHPAD_BASE         0x08000000UL
#define SCRATCHPAD_SIZE         (64UL * 1024UL)   /* 64 KB */

#define CORE0_TCM_BASE          0x08010000UL
#define CORE0_TCM_SIZE          0x2000UL          /* 8 KB */
#define CORE1_TCM_BASE          0x08012000UL
#define CORE1_TCM_SIZE          0x2000UL          /* 8 KB */

#endif /* MEMLAT_ADDRS_H */
