#include "dsp/basic_math_functions.h"

/**
  @ingroup groupMath
 */

/**
  @defgroup BasicDotProd Vector Dot Product

  Computes the dot product of two vectors.
  The vectors are multiplied element-by-element and then summed.

  <pre>
      sum = pSrcA[0]*pSrcB[0] + pSrcA[1]*pSrcB[1] + ... + pSrcA[blockSize-1]*pSrcB[blockSize-1]
  </pre>

  There are separate functions for floating-point, Q7, Q15, and Q31 data types.
 */

/**
  @addtogroup BasicDotProd
  @{
 */

/**
  @brief         Dot product of floating-point vectors.
  @param[in]     pSrcA      points to the first input vector.
  @param[in]     pSrcB      points to the second input vector.
  @param[in]     blockSize  number of samples in each vector.
  @param[out]    result     output result returned here.
 */


RISCV_DSP_ATTRIBUTE void riscv_dot_prod_f32(
  const float32_t * pSrcA,
  const float32_t * pSrcB,
        uint32_t blockSize,
        float32_t * result)
{
        float32_t sum = 0.0f;                          /* Temporary return variable */

#if defined(RISCV_MATH_VECTOR)
  size_t blkCnt = blockSize;                               /* Loop counter */
  size_t l;
  vfloat32m8_t v_A, v_B;
  vfloat32m8_t vsum;

  l = __riscv_vsetvlmax_e32m8();
  vsum = __riscv_vfmv_v_f_f32m8(0.0f, l);

  for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l)
  {
    v_A = __riscv_vle32_v_f32m8(pSrcA, l);
    pSrcA += l;
    v_B = __riscv_vle32_v_f32m8(pSrcB, l);
    pSrcB += l;
    vsum = __riscv_vfmacc_vv_f32m8(vsum, v_A, v_B, l);
  }
  l = __riscv_vsetvl_e32m8(1);
  vfloat32m1_t temp00 = __riscv_vfmv_v_f_f32m1(0.0f, l);
  l = __riscv_vsetvlmax_e32m8();
  temp00 = __riscv_vfredusum_vs_f32m8_f32m1(vsum, temp00, l);
  sum = __riscv_vfmv_f_s_f32m1_f32(temp00);
#else

  uint32_t blkCnt;                               /* Loop counter */

#if defined (RISCV_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  /* First part of the processing with loop unrolling. Compute 4 outputs at a time.
   ** a second loop below computes the remaining 1 to 3 samples. */
  while (blkCnt > 0U)
  {
    /* C = A[0]* B[0] + A[1]* B[1] + A[2]* B[2] + .....+ A[blockSize-1]* B[blockSize-1] */

    /* Calculate dot product and store result in a temporary buffer. */
    sum += (*pSrcA++) * (*pSrcB++);

    sum += (*pSrcA++) * (*pSrcB++);

    sum += (*pSrcA++) * (*pSrcB++);

    sum += (*pSrcA++) * (*pSrcB++);

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize & 0x3U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (RISCV_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A[0]* B[0] + A[1]* B[1] + A[2]* B[2] + .....+ A[blockSize-1]* B[blockSize-1] */

    /* Calculate dot product and store result in a temporary buffer. */
    sum += (*pSrcA++) * (*pSrcB++);

    /* Decrement loop counter */
    blkCnt--;
  }
#endif /* defined(RISCV_MATH_VECTOR) */
  /* Store result in destination buffer */
  *result = sum;
}

/**
  @} end of BasicDotProd group
 */
