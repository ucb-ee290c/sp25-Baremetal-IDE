#ifndef INTERPOLATION_FUNCTIONS_F16_H_
#define INTERPOLATION_FUNCTIONS_F16_H_

#include "riscv_math_types_f16.h"
#include "riscv_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if defined(RISCV_FLOAT16_SUPPORTED)

/**
 * @brief Instance structure for the half floating-point Linear Interpolate function.
 */
typedef struct
{
    uint32_t  nValues;        /**< nValues */
    float16_t x1;             /**< x1 */
    float16_t xSpacing;       /**< xSpacing */
    const float16_t *pYData;        /**< pointer to the table of Y values */
} riscv_linear_interp_instance_f16;

/**
 * @brief Instance structure for the floating-point bilinear interpolation function.
 */
typedef struct
{
    uint16_t  numRows;/**< number of rows in the data table. */
    uint16_t  numCols;/**< number of columns in the data table. */
    const float16_t *pData; /**< points to the data table. */
} riscv_bilinear_interp_instance_f16;

  /**
   * @addtogroup LinearInterpolate
   * @{
   */

    /**
   * @brief  Process function for the floating-point Linear Interpolation Function.
   * @param[in,out] S  is an instance of the floating-point Linear Interpolation structure
   * @param[in]     x  input sample to process
   * @return y processed output sample.
   */
  float16_t riscv_linear_interp_f16(
  const riscv_linear_interp_instance_f16 * S,
  float16_t x);

    /**
   * @} end of LinearInterpolate group
   */

/**
   * @addtogroup BilinearInterpolate
   * @{
   */

  /**
  * @brief  Floating-point bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate.
  * @param[in]     Y  interpolation coordinate.
  * @return out interpolated value.
  */
  float16_t riscv_bilinear_interp_f16(
  const riscv_bilinear_interp_instance_f16 * S,
  float16_t X,
  float16_t Y);


  /**
   * @} end of BilinearInterpolate group
   */
#endif /*defined(RISCV_FLOAT16_SUPPORTED)*/
#ifdef   __cplusplus
}
#endif

#endif /* ifndef _INTERPOLATION_FUNCTIONS_F16_H_ */
