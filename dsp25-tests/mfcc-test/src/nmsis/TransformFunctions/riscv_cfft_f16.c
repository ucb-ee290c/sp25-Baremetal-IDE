#include "dsp/transform_functions_f16.h"
#include "riscv_common_tables_f16.h"



#if defined(RISCV_FLOAT16_SUPPORTED)

extern void riscv_bitreversal_16(
        uint16_t * pSrc,
  const uint16_t bitRevLen,
  const uint16_t * pBitRevTable);


extern void riscv_cfft_radix4by2_f16(
    float16_t * pSrc,
    uint32_t fftLen,
    const float16_t * pCoef);

extern void riscv_radix4_butterfly_f16(
        float16_t * pSrc,
        uint16_t fftLen,
  const float16_t * pCoef,
        uint16_t twidCoefModifier);

/**
  @addtogroup ComplexFFTF16
  @{
 */

/**
  @brief         Processing function for the floating-point complex FFT.
  @param[in]     S              points to an instance of the floating-point CFFT structure
  @param[in,out] p1             points to the complex data buffer of size <code>2*fftLen</code>. Processing occurs in-place
  @param[in]     ifftFlag       flag that selects transform direction
                   - value = 0: forward transform
                   - value = 1: inverse transform
  @param[in]     bitReverseFlag flag that enables / disables bit reversal of output
                   - value = 0: disables bit reversal of output
                   - value = 1: enables bit reversal of output
 */

RISCV_DSP_ATTRIBUTE void riscv_cfft_f16(
    const riscv_cfft_instance_f16 * S,
    float16_t * p1,
    uint8_t ifftFlag,
    uint8_t bitReverseFlag)
{
    uint32_t  L = S->fftLen;
    float16_t invL;

    if (ifftFlag == 1U)
    {
        /*  Conjugate input data  */
#if defined(RISCV_MATH_VECTOR_F16)
        uint32_t blkCnt = L;
        size_t vl;
        float16_t *pImag = p1 + 1;
        ptrdiff_t stride = (ptrdiff_t)(2U * sizeof(float16_t));
        while ((vl = __riscv_vsetvl_e16m8(blkCnt)) > 0)
        {
            vfloat16m8_t vI = __riscv_vlse16_v_f16m8(pImag, stride, vl);
            vI = __riscv_vfmul_vf_f16m8(vI, -1.0f16, vl);
            __riscv_vsse16_v_f16m8(pImag, stride, vI, vl);
            pImag += (2U * (uint32_t)vl);
            blkCnt -= (uint32_t)vl;
        }
#else
        uint32_t l;
        float16_t *pSrc;
        pSrc = p1 + 1;
        for(l=0; l<L; l++)
        {
            *pSrc = -(_Float16)*pSrc;
            pSrc += 2;
        }
#endif
    }

    switch (L)
    {

        case 16:
        case 64:
        case 256:
        case 1024:
        case 4096:
        riscv_radix4_butterfly_f16  (p1, L, (float16_t*)S->pTwiddle, 1U);
        break;

        case 32:
        case 128:
        case 512:
        case 2048:
        riscv_cfft_radix4by2_f16  ( p1, L, (float16_t*)S->pTwiddle);
        break;

    }

    if ( bitReverseFlag )
        riscv_bitreversal_16((uint16_t*)p1, S->bitRevLength,(uint16_t*)S->pBitRevTable);

    if (ifftFlag == 1U)
    {
        invL = 1.0f16/(_Float16)L;
        /*  Conjugate and scale output data */
#if defined(RISCV_MATH_VECTOR_F16)
        uint32_t blkCnt = L;
        size_t vl;
        float16_t *pReal = p1;
        float16_t *pImag = p1 + 1;
        ptrdiff_t stride = (ptrdiff_t)(2U * sizeof(float16_t));
        const float16_t negInvL = -invL;
        while ((vl = __riscv_vsetvl_e16m8(blkCnt)) > 0)
        {
            vfloat16m8_t vR = __riscv_vlse16_v_f16m8(pReal, stride, vl);
            vfloat16m8_t vI = __riscv_vlse16_v_f16m8(pImag, stride, vl);
            vR = __riscv_vfmul_vf_f16m8(vR, invL, vl);
            vI = __riscv_vfmul_vf_f16m8(vI, negInvL, vl);
            __riscv_vsse16_v_f16m8(pReal, stride, vR, vl);
            __riscv_vsse16_v_f16m8(pImag, stride, vI, vl);
            pReal += (2U * (uint32_t)vl);
            pImag += (2U * (uint32_t)vl);
            blkCnt -= (uint32_t)vl;
        }
#else
        uint32_t l;
        float16_t *pSrc;
        pSrc = p1;
        for(l=0; l<L; l++)
        {
            *pSrc++ *=   (_Float16)invL ;
            *pSrc  = -(_Float16)(*pSrc) * (_Float16)invL;
            pSrc++;
        }
#endif
    }
}
#endif /* if defined(RISCV_FLOAT16_SUPPORTED) */

/**
  @} end of ComplexFFTF16 group
 */
