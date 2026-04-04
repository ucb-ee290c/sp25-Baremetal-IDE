#ifndef RISCV_CONST_STRUCTS_H
#define RISCV_CONST_STRUCTS_H

#include "riscv_math_types.h"
#include "riscv_common_tables.h"
#include "dsp/transform_functions.h"

#if defined (RISCV_FLOAT16_SUPPORTED)
#include "riscv_const_structs_f16.h"
#endif /* defined (RISCV_FLOAT16_SUPPORTED) */

#ifdef   __cplusplus
extern "C"
{
#endif
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len16;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len32;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len64;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len128;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len256;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len512;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len1024;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len2048;
   extern const riscv_cfft_instance_f64 riscv_cfft_sR_f64_len4096;

   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len16;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len32;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len64;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len128;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len256;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len512;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len1024;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len2048;
   extern const riscv_cfft_instance_f32 riscv_cfft_sR_f32_len4096;

   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len16;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len32;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len64;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len128;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len256;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len512;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len1024;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len2048;
   extern const riscv_cfft_instance_q31 riscv_cfft_sR_q31_len4096;

   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len16;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len32;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len64;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len128;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len256;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len512;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len1024;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len2048;
   extern const riscv_cfft_instance_q15 riscv_cfft_sR_q15_len4096;

#ifdef   __cplusplus
}
#endif

#endif

