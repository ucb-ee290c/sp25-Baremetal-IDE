#include "dsp/complex_math_functions.h"

/**
  @ingroup groupCmplxMath
 */

/**
  @defgroup cmplx_mag Complex Magnitude

  Computes the magnitude of the elements of a complex data vector.

  The <code>pSrc</code> points to the source data and
  <code>pDst</code> points to the where the result should be written.
  <code>numSamples</code> specifies the number of complex samples
  in the input array and the data is stored in an interleaved fashion
  (real, imag, real, imag, ...).
  The input array has a total of <code>2*numSamples</code> values;
  the output array has a total of <code>numSamples</code> values.

  The underlying algorithm is used:

  <pre>
  for (n = 0; n < numSamples; n++) {
      pDst[n] = sqrt(pSrc[(2*n)+0]^2 + pSrc[(2*n)+1]^2);
  }
  </pre>

  There are separate functions for floating-point, Q15, and Q31 data types.
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


RISCV_DSP_ATTRIBUTE void riscv_cmplx_mag_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t numSamples)
{
#if defined(RISCV_MATH_VECTOR)
  uint32_t blkCnt = numSamples;                               /* Loop counter */
  size_t l;
  ptrdiff_t bstride = 8;
  vfloat32m8_t v_R, v_I;
  vfloat32m8_t v_sum;

  for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l)
  {
    v_R = __riscv_vlse32_v_f32m8(pSrc, bstride, l);
    v_I = __riscv_vlse32_v_f32m8(pSrc + 1, bstride, l);
    pSrc += l * 2;
    v_sum = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vv_f32m8(v_R, v_R, l), __riscv_vfmul_vv_f32m8(v_I, v_I, l), l);
    __riscv_vse32_v_f32m8(pDst, __riscv_vfsqrt_v_f32m8(v_sum, l), l);
    pDst += l;
  }
#else
  uint32_t blkCnt;                               /* loop counter */
  float32_t real, imag;                      /* Temporary variables to hold input values */


#if defined (RISCV_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = numSamples >> 2U;

  while (blkCnt > 0U)
  {
    /* C[0] = sqrt(A[0] * A[0] + A[1] * A[1]) */

    real = *pSrc++;
    imag = *pSrc++;

    /* store result in destination buffer. */
    riscv_sqrt_f32((real * real) + (imag * imag), pDst++);

    real = *pSrc++;
    imag = *pSrc++;
    riscv_sqrt_f32((real * real) + (imag * imag), pDst++);

    real = *pSrc++;
    imag = *pSrc++;
    riscv_sqrt_f32((real * real) + (imag * imag), pDst++);

    real = *pSrc++;
    imag = *pSrc++;
    riscv_sqrt_f32((real * real) + (imag * imag), pDst++);

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
    riscv_sqrt_f32((real * real) + (imag * imag), pDst++);

    /* Decrement loop counter */
    blkCnt--;
  }
#endif /* defined(RISCV_MATH_VECTOR) */
}

/**
  @} end of cmplx_mag group
 */
