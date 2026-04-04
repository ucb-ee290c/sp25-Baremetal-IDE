#include "dsp/fast_math_functions.h"
#include "riscv_common_tables.h"

#include <stdlib.h>

/**
  @ingroup groupFastMath
 */

/**
  @defgroup divide Fixed point division

 */

/**
  @addtogroup divide
  @{
 */

/**
  @brief         Fixed point division
  @param[in]     numerator    Numerator
  @param[in]     denominator  Denominator
  @param[out]    quotient     Quotient value normalized between -1.0 and 1.0
  @param[out]    shift        Shift left value to get the unnormalized quotient
  @return        error status

  When dividing by 0, an error RISCV_MATH_NANINF is returned. And the quotient is forced
  to the saturated negative or positive value.
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_divide_q15(q15_t numerator,
  q15_t denominator,
  q15_t *quotient,
  int16_t *shift)
{
  int16_t sign=0;
  q31_t temp;
  int16_t shiftForNormalizing;

  *shift = 0;

  sign = (numerator<0) ^ (denominator<0);

  if (denominator == 0)
  {
     if (sign)
     {
        *quotient = -32768;
     }
     else
     {
        *quotient = 32767;
     }
     return(RISCV_MATH_NANINF);
  }

  riscv_abs_q15(&numerator,&numerator,1);
  riscv_abs_q15(&denominator,&denominator,1);
  
  temp = ((q31_t)numerator << 15) / ((q31_t)denominator);

  shiftForNormalizing= 17 - __CLZ(temp);
  if (shiftForNormalizing > 0)
  {
     *shift = shiftForNormalizing;
     temp = temp >> shiftForNormalizing;
  }

  if (sign)
  {
    temp = -temp;
  }

  *quotient=temp;

  return(RISCV_MATH_SUCCESS);
}

/**
  @} end of divide group
 */
