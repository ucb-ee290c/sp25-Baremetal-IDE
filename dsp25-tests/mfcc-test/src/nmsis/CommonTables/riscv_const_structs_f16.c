#include "riscv_math_types_f16.h"

#if defined(RISCV_FLOAT16_SUPPORTED)

#include "riscv_const_structs_f16.h"




/* Floating-point structs */


/* 

Those structures cannot be used to initialize the MVE version of the FFT F32 instances.
So they are not compiled when MVE is defined.

For the MVE version, the new riscv_cfft_init_f16 must be used.


*/

 
const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len16 = {
  16, twiddleCoefF16_16, riscvBitRevIndexTable_fixed_16, RISCVBITREVINDEXTABLE_FIXED_16_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len32 = {
  32, twiddleCoefF16_32, riscvBitRevIndexTable_fixed_32, RISCVBITREVINDEXTABLE_FIXED_32_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len64 = {
  64, twiddleCoefF16_64, riscvBitRevIndexTable_fixed_64, RISCVBITREVINDEXTABLE_FIXED_64_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len128 = {
  128, twiddleCoefF16_128, riscvBitRevIndexTable_fixed_128, RISCVBITREVINDEXTABLE_FIXED_128_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len256 = {
  256, twiddleCoefF16_256, riscvBitRevIndexTable_fixed_256, RISCVBITREVINDEXTABLE_FIXED_256_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len512 = {
  512, twiddleCoefF16_512, riscvBitRevIndexTable_fixed_512, RISCVBITREVINDEXTABLE_FIXED_512_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len1024 = {
  1024, twiddleCoefF16_1024, riscvBitRevIndexTable_fixed_1024, RISCVBITREVINDEXTABLE_FIXED_1024_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len2048 = {
  2048, twiddleCoefF16_2048, riscvBitRevIndexTable_fixed_2048, RISCVBITREVINDEXTABLE_FIXED_2048_TABLE_LENGTH
};

const riscv_cfft_instance_f16 riscv_cfft_sR_f16_len4096 = {
  4096, twiddleCoefF16_4096, riscvBitRevIndexTable_fixed_4096, RISCVBITREVINDEXTABLE_FIXED_4096_TABLE_LENGTH
};

#endif /* defined (RISCV_FLOAT16_SUPPORTED) */
