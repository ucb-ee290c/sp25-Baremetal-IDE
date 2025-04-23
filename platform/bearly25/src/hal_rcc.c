#include "hal_rcc.h"

/**
 * \brief           Sets all clock domains to the same index `clksrc`.
 * \param[in]       clksel: Clock selector instance
 * \param[in]       clksrc: Clock selector input to select.
 */
void set_all_clocks(ClockSel_Type* clksel, ClockSel_Opts clksrc) {
  clksel->UNCORE = clksrc;
  clksel->TILE0  = clksrc;
  clksel->TILE1  = clksrc;
  clksel->CLKTAP = clksrc;
}
