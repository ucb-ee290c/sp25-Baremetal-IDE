#ifndef TINYSPEECH_MC_BENCH_CONFIG_H
#define TINYSPEECH_MC_BENCH_CONFIG_H

#include <stdint.h>

#ifndef TINYSPEECH_MC_TARGET_FREQUENCY_HZ
#define TINYSPEECH_MC_TARGET_FREQUENCY_HZ 500000000ULL
#endif

// 0: single-frequency mode
// 1: iterate over TINYSPEECH_MC_PLL_FREQ_LIST
#ifndef TINYSPEECH_MC_ENABLE_PLL_SWEEP
#define TINYSPEECH_MC_ENABLE_PLL_SWEEP 0
#endif

#ifndef TINYSPEECH_MC_PLL_SWEEP_SLEEP_MS
#define TINYSPEECH_MC_PLL_SWEEP_SLEEP_MS 10000u
#endif

// Comma-separated list used when TINYSPEECH_MC_ENABLE_PLL_SWEEP=1
// #define TINYSPEECH_MC_PLL_FREQ_LIST 50000000ULL, 150000000ULL, 250000000ULL
#ifndef TINYSPEECH_MC_PLL_FREQ_LIST
#define TINYSPEECH_MC_PLL_FREQ_LIST \
  50000000ULL, \
  150000000ULL, \
  250000000ULL, \
  350000000ULL
#endif

#ifndef TINYSPEECH_MC_ENABLE_MULTICORE
#define TINYSPEECH_MC_ENABLE_MULTICORE 1
#endif

// Output-channel split points for 2-core fixed-shape INT8 kernels.
#ifndef TINYSPEECH_MC_CONV2_OC_SPLIT
#define TINYSPEECH_MC_CONV2_OC_SPLIT 24
#endif

#ifndef TINYSPEECH_MC_CONV3_OC_SPLIT
#define TINYSPEECH_MC_CONV3_OC_SPLIT 48
#endif

// Optional memory placement hooks (disabled by default for portability).
// When enabled, tiny hot buffers can be staged in scratchpad/TCM.
#ifndef TINYSPEECH_MC_USE_SCRATCHPAD_FC
#define TINYSPEECH_MC_USE_SCRATCHPAD_FC 0
#endif

// 1: map hot shared fixed-path int8 buffers into scratchpad (64KB region)
//    pad2, pad3, gap3(hart0/shared), and w2_pack16.
#ifndef TINYSPEECH_MC_USE_SCRATCHPAD_SHARED
#define TINYSPEECH_MC_USE_SCRATCHPAD_SHARED 1
#endif

// 1: use core-private TCM for tiny private buffers.
//    current use: hart1 private gap3 accumulator + optional FC weights in core0 TCM.
#ifndef TINYSPEECH_MC_USE_TCM_PRIVATE
#define TINYSPEECH_MC_USE_TCM_PRIVATE 1
#endif

#ifndef TINYSPEECH_MC_SCRATCHPAD_BYTES
#define TINYSPEECH_MC_SCRATCHPAD_BYTES (64u * 1024u)
#endif

#ifndef TINYSPEECH_MC_TCM_BYTES
#define TINYSPEECH_MC_TCM_BYTES (8u * 1024u)
#endif

// Scratchpad layout offsets for fixed-path hot buffers.
#ifndef TINYSPEECH_MC_SCRATCH_OFFSET_PAD2
#define TINYSPEECH_MC_SCRATCH_OFFSET_PAD2 0u
#endif

#ifndef TINYSPEECH_MC_SCRATCH_OFFSET_PAD3
#define TINYSPEECH_MC_SCRATCH_OFFSET_PAD3 12288u
#endif

#ifndef TINYSPEECH_MC_SCRATCH_OFFSET_GAP3
#define TINYSPEECH_MC_SCRATCH_OFFSET_GAP3 18432u
#endif

#ifndef TINYSPEECH_MC_SCRATCH_OFFSET_W2PACK16
#define TINYSPEECH_MC_SCRATCH_OFFSET_W2PACK16 20480u
#endif

// Core-local TCM layout offsets for tiny private buffers.
#ifndef TINYSPEECH_MC_CORE1_TCM_OFFSET_GAP3_PRIV
#define TINYSPEECH_MC_CORE1_TCM_OFFSET_GAP3_PRIV 0u
#endif

#ifndef TINYSPEECH_MC_CORE0_TCM_OFFSET_FC
#define TINYSPEECH_MC_CORE0_TCM_OFFSET_FC 0u
#endif

#ifndef TINYSPEECH_MC_SCRATCHPAD_BASE
#define TINYSPEECH_MC_SCRATCHPAD_BASE 0x08000000UL
#endif

#ifndef TINYSPEECH_MC_CORE0_TCM_BASE
#define TINYSPEECH_MC_CORE0_TCM_BASE 0x08010000UL
#endif

#ifndef TINYSPEECH_MC_CORE1_TCM_BASE
#define TINYSPEECH_MC_CORE1_TCM_BASE 0x08012000UL
#endif

#endif
