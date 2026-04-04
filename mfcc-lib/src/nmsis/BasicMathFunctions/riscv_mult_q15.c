#include "dsp/basic_math_functions.h"

/**
  @ingroup groupMath
 */

/**
  @addtogroup BasicMult
  @{
 */

/**
  @brief         Q15 vector multiplication
  @param[in]     pSrcA      points to first input vector
  @param[in]     pSrcB      points to second input vector
  @param[out]    pDst       points to output vector
  @param[in]     blockSize  number of samples in each vector

  @par           Scaling and Overflow Behavior
                   The function uses saturating arithmetic.
                   Results outside of the allowable Q15 range [0x8000 0x7FFF] are saturated.
 */
RISCV_DSP_ATTRIBUTE void riscv_mult_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t blockSize)
{
  uint32_t blkCnt;                               /* Loop counter */

#if defined(RISCV_MATH_VECTOR)
  blkCnt = blockSize;                               /* Loop counter */
  size_t l;
  vint16m8_t vx, vy;

  for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l)
  {
    vx = __riscv_vle16_v_i16m8(pSrcA, l);
    pSrcA += l;
    vy = __riscv_vle16_v_i16m8(pSrcB, l);
    pSrcB += l;
#if defined(__RISCV_VXRM_RNU)
    __riscv_vse16_v_i16m8(pDst, __riscv_vsmul_vv_i16m8(vx, vy, __RISCV_VXRM_RNU, l), l);
#else
    __riscv_vse16_v_i16m8(pDst, __riscv_vsmul_vv_i16m8(vx, vy, l), l);
#endif
    pDst += l;
  }
#else

#if defined (RISCV_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A * B */

#if defined (RISCV_MATH_DSP)
#if __RISCV_XLEN == 64
    write_q15x4_ia(&pDst, __RV_KHM16(read_q15x4_ia((q15_t **)& pSrcA), read_q15x4_ia((q15_t**)&pSrcB)));
#else
#ifdef NUCLEI_DSP_N1
    write_q15x4_ia(&pDst, __RV_DKHM16(read_q15x4_ia((q15_t **)& pSrcA), read_q15x4_ia((q15_t**)&pSrcB)));
#else
    write_q15x2_ia(&pDst, __RV_KHM16(read_q15x2_ia((q15_t **)& pSrcA), read_q15x2_ia((q15_t**)&pSrcB)));
    write_q15x2_ia(&pDst, __RV_KHM16(read_q15x2_ia((q15_t **)& pSrcA), read_q15x2_ia((q15_t**)&pSrcB)));
#endif /* NUCLEI_DSP_N1 */
#endif /* __RISCV_XLEN == 64 */
#else
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
#endif /* RISCV_MATH_DSP */

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

    /* Multiply inputs and store result in destination buffer. */
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);

    /* Decrement loop counter */
    blkCnt--;
  }
#endif /* defined(RISCV_MATH_VECTOR) */
}

/**
  @} end of BasicMult group
 */
