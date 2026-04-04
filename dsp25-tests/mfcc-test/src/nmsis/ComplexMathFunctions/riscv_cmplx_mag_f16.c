#include "dsp/complex_math_functions_f16.h"

#if defined(RISCV_FLOAT16_SUPPORTED)
/**
  @ingroup groupCmplxMath
 */



/**
  @addtogroup cmplx_mag
  @{
 */

/**
  @brief         Floating-point complex magnitude.
  @param[in]     pSrc        points to input vector
  @param[out]    pDst        points to output vector
  @param[in]     numSamples  number of samples in each vector
 */

RISCV_DSP_ATTRIBUTE void riscv_cmplx_mag_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t numSamples)
{
#if defined(RISCV_MATH_VECTOR_F16)
  uint32_t blkCnt = numSamples;                               /* Loop counter */
  size_t l;
  ptrdiff_t bstride = 4;
  vfloat16m8_t v_R, v_I;
  vfloat16m8_t v_sum;

  for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l)
  {
    v_R = __riscv_vlse16_v_f16m8(pSrc, bstride, l);
    v_I = __riscv_vlse16_v_f16m8(pSrc + 1, bstride, l);
    pSrc += l * 2;
    v_sum = __riscv_vfadd_vv_f16m8(__riscv_vfmul_vv_f16m8(v_R, v_R, l), __riscv_vfmul_vv_f16m8(v_I, v_I, l), l);
    __riscv_vse16_v_f16m8(pDst, __riscv_vfsqrt_v_f16m8(v_sum, l), l);
    pDst += l;
  }
#else
  uint32_t blkCnt;                               /* loop counter */
  _Float16 real, imag;                      /* Temporary variables to hold input values */

#if defined (RISCV_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = numSamples >> 2U;

  while (blkCnt > 0U)
  {
    /* C[0] = sqrt(A[0] * A[0] + A[1] * A[1]) */

    real = *pSrc++;
    imag = *pSrc++;

    /* store result in destination buffer. */
    riscv_sqrt_f16((real * real) + (imag * imag), pDst++);

    real = *pSrc++;
    imag = *pSrc++;
    riscv_sqrt_f16((real * real) + (imag * imag), pDst++);

    real = *pSrc++;
    imag = *pSrc++;
    riscv_sqrt_f16((real * real) + (imag * imag), pDst++);

    real = *pSrc++;
    imag = *pSrc++;
    riscv_sqrt_f16((real * real) + (imag * imag), pDst++);

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = numSamples & 0x3U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = numSamples;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C[0] = sqrt(A[0] * A[0] + A[1] * A[1]) */

    real = *pSrc++;
    imag = *pSrc++;

    /* store result in destination buffer. */
    riscv_sqrt_f16((real * real) + (imag * imag), pDst++);

    /* Decrement loop counter */
    blkCnt--;
  }
#endif /* defined(RISCV_MATH_VECTOR_F16) */
}

/**
  @} end of cmplx_mag group
 */

#endif /* #if defined(RISCV_FLOAT16_SUPPORTED) */
