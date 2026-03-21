#ifndef RISCV_COMMON_TABLES_F16_H
#define RISCV_COMMON_TABLES_F16_H

#include "riscv_math_types_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif


  /* F16 */
  #if defined(RISCV_FLOAT16_SUPPORTED)
    extern const float16_t twiddleCoefF16_16[32];

    extern const float16_t twiddleCoefF16_32[64];

    extern const float16_t twiddleCoefF16_64[128];

    extern const float16_t twiddleCoefF16_128[256];

    extern const float16_t twiddleCoefF16_256[512];

    extern const float16_t twiddleCoefF16_512[1024];

    extern const float16_t twiddleCoefF16_1024[2048];

    extern const float16_t twiddleCoefF16_2048[4096];

    extern const float16_t twiddleCoefF16_4096[8192];
    #define twiddleCoefF16 twiddleCoefF16_4096

    extern const float16_t twiddleCoefF16_rfft_32[32];

    extern const float16_t twiddleCoefF16_rfft_64[64];

    extern const float16_t twiddleCoefF16_rfft_128[128];

    extern const float16_t twiddleCoefF16_rfft_256[256];

    extern const float16_t twiddleCoefF16_rfft_512[512];

    extern const float16_t twiddleCoefF16_rfft_1024[1024];

    extern const float16_t twiddleCoefF16_rfft_2048[2048];

    extern const float16_t twiddleCoefF16_rfft_4096[4096];

  #endif /* #if defined(RISCV_FLOAT16_SUPPORTED) */

#ifdef   __cplusplus
}
#endif

#endif /*  _RISCV_COMMON_TABLES_F16_H */
