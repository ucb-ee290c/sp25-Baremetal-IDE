#ifndef RISCV_CONST_STRUCTS_F16_H
#define RISCV_CONST_STRUCTS_F16_H

#include "riscv_math_types_f16.h"
#include "riscv_common_tables.h"
#include "riscv_common_tables_f16.h"
#include "dsp/transform_functions_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if defined(RISCV_FLOAT16_SUPPORTED)
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len16;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len32;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len64;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len128;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len256;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len512;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len1024;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len2048;
   extern const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len4096;
#endif

#ifdef   __cplusplus
}
#endif

#endif
