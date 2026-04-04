#ifndef COMPLEX_MATH_FUNCTIONS_F16_H_
#define COMPLEX_MATH_FUNCTIONS_F16_H_

#include "riscv_math_types_f16.h"
#include "riscv_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"
#include "dsp/fast_math_functions_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if defined(RISCV_FLOAT16_SUPPORTED)

 /**
   * @brief  Floating-point complex conjugate.
   * @param[in]  pSrc        points to the input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void riscv_cmplx_conj_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t numSamples);

 /**
   * @brief  Floating-point complex magnitude squared
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void riscv_cmplx_mag_squared_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t numSamples);

  /**
   * @brief  Floating-point complex magnitude
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void riscv_cmplx_mag_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t numSamples);

  /**
   * @brief  Floating-point complex dot product
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[in]  numSamples  number of complex samples in each vector
   * @param[out] realResult  real part of the result returned here
   * @param[out] imagResult  imaginary part of the result returned here
   */
  void riscv_cmplx_dot_prod_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        uint32_t numSamples,
        float16_t * realResult,
        float16_t * imagResult);

   /**
   * @brief  Floating-point complex-by-real multiplication
   * @param[in]  pSrcCmplx   points to the complex input vector
   * @param[in]  pSrcReal    points to the real input vector
   * @param[out] pCmplxDst   points to the complex output vector
   * @param[in]  numSamples  number of samples in each vector
   */
  void riscv_cmplx_mult_real_f16(
  const float16_t * pSrcCmplx,
  const float16_t * pSrcReal,
        float16_t * pCmplxDst,
        uint32_t numSamples);

  /**
   * @brief  Floating-point complex-by-complex multiplication
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void riscv_cmplx_mult_cmplx_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
        uint32_t numSamples);

#endif /*defined(RISCV_FLOAT16_SUPPORTED)*/
#ifdef   __cplusplus
}
#endif

#endif /* ifndef _COMPLEX_MATH_FUNCTIONS_F16_H_ */
