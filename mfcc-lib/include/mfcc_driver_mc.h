#ifndef MFCC_DRIVER_MC_H
#define MFCC_DRIVER_MC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mfcc_driver.h"

mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f32_mc(mfcc_driver_t *ctx,
                                                          const float32_t *input,
                                                          float32_t *output,
                                                          uint64_t *cycles);

#if defined(RISCV_FLOAT16_SUPPORTED)
mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f16_mc(mfcc_driver_t *ctx,
                                                          const float32_t *input,
                                                          float16_t *output,
                                                          uint64_t *cycles);
#endif

void mfcc_driver_mc_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
