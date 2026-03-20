#include "dsp/basic_math_functions_f16.h"

/**
  @ingroup groupMath
 */


/**
  @addtogroup BasicMult
  @{
 */

/**
  @brief         Floating-point vector multiplication.
  @param[in]     pSrcA      points to the first input vector.
  @param[in]     pSrcB      points to the second input vector.
  @param[out]    pDst       points to the output vector.
  @param[in]     blockSize  number of samples in each vector.
 */


#if defined(RISCV_FLOAT16_SUPPORTED)
RISCV_DSP_ATTRIBUTE void riscv_mult_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
        uint32_t blockSize)
{
    uint32_t blkCnt;                               /* Loop counter */

#if defined(RISCV_MATH_VECTOR_F16)
  blkCnt = blockSize;                               /* Loop counter */
  size_t l;
  vfloat16m8_t vx, vy;

  for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
    vx = __riscv_vle16_v_f16m8(pSrcA, l);
    pSrcA += l;
    vy = __riscv_vle16_v_f16m8(pSrcB, l);
    pSrcB += l;
    __riscv_vse16_v_f16m8(pDst, __riscv_vfmul_vv_f16m8(vx, vy, l), l);
    pDst += l;
  }
#else
#if defined (RISCV_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A * B */

    /* Multiply inputs and store result in destination buffer. */
    *pDst++ = (_Float16)(*pSrcA++) * (_Float16)(*pSrcB++);

    *pDst++ = (_Float16)(*pSrcA++) * (_Float16)(*pSrcB++);

    *pDst++ = (_Float16)(*pSrcA++) * (_Float16)(*pSrcB++);

    *pDst++ = (_Float16)(*pSrcA++) * (_Float16)(*pSrcB++);

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
    /* C = A * B */

    /* Multiply input and store result in destination buffer. */
    *pDst++ = (_Float16)(*pSrcA++) * (_Float16)(*pSrcB++);

    /* Decrement loop counter */
    blkCnt--;
  }
#endif /* defined(RISCV_MATH_VECTOR_F16) */
}
#endif

/**
  @} end of BasicMult group
 */
