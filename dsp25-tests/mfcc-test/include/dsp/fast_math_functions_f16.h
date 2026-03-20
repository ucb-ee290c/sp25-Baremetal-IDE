#ifndef FAST_MATH_FUNCTIONS_F16_H_
#define FAST_MATH_FUNCTIONS_F16_H_

#include "riscv_math_types_f16.h"
#include "riscv_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"

/* For sqrt_f32 */
#include "dsp/fast_math_functions.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if defined(RISCV_FLOAT16_SUPPORTED)

 /**
   * @addtogroup SQRT
   * @{
   */

/**
  @brief         Floating-point square root function.
  @param[in]     in    input value
  @param[out]    pOut  square root of input value
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : input value is positive
                   - \ref RISCV_MATH_ARGUMENT_ERROR : input value is negative; *pOut is set to 0
 */
__STATIC_FORCEINLINE riscv_status riscv_sqrt_f16(
  float16_t in,
  float16_t * pOut)
  {
    float32_t r;
    riscv_status status;
    status=riscv_sqrt_f32((float32_t)in,&r);
    *pOut=(float16_t)r;
    return(status);
  }


/**
  @} end of SQRT group
 */
  
/**
  @brief         Floating-point vector of log values.
  @param[in]     pSrc       points to the input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
 */
  void riscv_vlog_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t blockSize);

/**
  @brief         Floating-point vector of exp values.
  @param[in]     pSrc       points to the input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
 */
  void riscv_vexp_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t blockSize);

  /**
  @brief         Floating-point vector of inverse values.
  @param[in]     pSrc       points to the input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
 */
  void riscv_vinverse_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t blockSize);

  /**
     @brief  Arc tangent in radian of y/x using sign of x and y to determine right quadrant.
     @param[in]   y  y coordinate
     @param[in]   x  x coordinate
     @param[out]  result  Result
     @return  error status.
   */
  riscv_status riscv_atan2_f16(float16_t y,float16_t x,float16_t *result);

#endif /*defined(RISCV_FLOAT16_SUPPORTED)*/
#ifdef   __cplusplus
}
#endif

#endif /* ifndef _FAST_MATH_FUNCTIONS_F16_H_ */
