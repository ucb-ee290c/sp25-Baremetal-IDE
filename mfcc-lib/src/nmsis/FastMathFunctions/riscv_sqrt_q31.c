#include "dsp/fast_math_functions.h"
#include "riscv_common_tables.h"

#define Q28QUARTER 0x20000000

/**
  @ingroup groupFastMath
 */

/**
  @addtogroup SQRT
  @{
 */

/**
  @brief         Q31 square root function.
  @param[in]     in    input value.  The range of the input value is [0 +1) or 0x00000000 to 0x7FFFFFFF
  @param[out]    pOut  points to square root of input value
  @return        execution status
                   - \ref RISCV_MATH_SUCCESS        : input value is positive
                   - \ref RISCV_MATH_ARGUMENT_ERROR : input value is negative; *pOut is set to 0
 */

RISCV_DSP_ATTRIBUTE riscv_status riscv_sqrt_q31(
  q31_t in,
  q31_t * pOut)
{
  q31_t number, var1, signBits1, temp;

  number = in;

  /* If the input is a positive number then compute the signBits. */
  if (number > 0)
  {
    signBits1 = __CLZ(number) - 1;

    /* Shift by the number of signBits1 */
    if ((signBits1 % 2) == 0)
    {
      number = number << signBits1;
    }
    else
    {
      number = number << (signBits1 - 1);
    }

    /* Start value for 1/sqrt(x) for the Newton iteration */
    var1 = sqrt_initial_lut_q31[(number>> 26) - (Q28QUARTER >> 26)];

    /* 0.5 var1 * (3 - number * var1 * var1) */

    /* 1st iteration */

    temp = ((q63_t) var1 * var1) >> 28;
    temp = ((q63_t) number * temp) >> 31;
    temp = 0x30000000 - temp;
    var1 = ((q63_t) var1 * temp) >> 29;


    /* 2nd iteration */
    temp = ((q63_t) var1 * var1) >> 28;
    temp = ((q63_t) number * temp) >> 31;
    temp = 0x30000000 - temp;
    var1 = ((q63_t) var1 * temp) >> 29;

    /* 3nd iteration */
    temp = ((q63_t) var1 * var1) >> 28;
    temp = ((q63_t) number * temp) >> 31;
    temp = 0x30000000 - temp;
    var1 = ((q63_t) var1 * temp) >> 29;

    /* Multiply the inverse square root with the original value */
    var1 = ((q31_t) (((q63_t) number * var1) >> 28));

    /* Shift the output down accordingly */
    if ((signBits1 % 2) == 0)
    {
      var1 = var1 >> (signBits1 / 2);
    }
    else
    {
      var1 = var1 >> ((signBits1 - 1) / 2);
    }
    *pOut = var1;

    return (RISCV_MATH_SUCCESS);
  }
  /* If the number is a negative number then store zero as its square root value */
  else
  {
    *pOut = 0;

    if (number==0)
    {
       return (RISCV_MATH_SUCCESS);
    }
    else
    {
       return (RISCV_MATH_ARGUMENT_ERROR);
    }
  }
}

/**
  @} end of SQRT group
 */
