
#include "pll.h"

/**
 * \brief           Configures the PLL to a specific frequency ratio.
 *                  PLL(1|2) = clk_in_ref * (ratio + (fraction/(2^24))
 * \param[in]       pll: Intel Ring PLL instance
 * \param[in]       ratio: Integer multiplication ratio component
 * \param[in]       fraction: Fractional multiplication ratio component
 */
void configure_pll(PLL_Type* pll, uint32_t ratio, uint32_t fraction) {
  pll->PLLEN = 0;
  pll->MDIV_RATIO = 1;
  pll->RATIO = ratio;
  pll->FRACTION = fraction;
  pll->ZDIV0_RATIO = 1;
  pll->ZDIV1_RATIO = 1;
  pll->LDO_ENABLE = 1;
  pll->PLLEN = 1;
  pll->POWERGOOD_VNN = 1;
  pll->PLLFWEN_B = 1;
}
