#ifndef __CORE_FEATURE_BITMANIP__
#define __CORE_FEATURE_BITMANIP__

/*
 * Bitmanipulation Feature Configuration Macro:
 * 1. __BITMANIP_PRESENT:  Define whether Bitmanipulation Unit is present or not
 *   * 0: Not present
 *   * 1: Present
 */
#ifdef __cplusplus
 extern "C" {
#endif

#include "core_feature_base.h"

#if defined(__BITMANIP_PRESENT) && (__BITMANIP_PRESENT == 1)

/* ###########################  CPU Bitmanipulation Intrinsic Functions ########################### */
/**
 * \defgroup NMSIS_Core_Bitmanip_Intrinsic   Intrinsic Functions for Bitmanipulation Instructions
 * \ingroup  NMSIS_Core
 * \brief    Functions that generate RISC-V Bitmanipulation instructions.
 * \details
 *
 * RISC-V Bitmanipulation Intrinsic APIs are provided directly through compiler generated intrinsic function.
 *
 * This intrinsic function support in compiler is introduced in nuclei riscv gcc 10.2.
 *
 * API header file can be found in lib/gcc/riscv-nuclei-elf/<gcc_ver>/include/rvintrin.h
 *
 * For Nuclei GCC 13/Clang 17 and later version, this intrinsic header no longer existed, please take care.
 *
 *   @{
 */
/** @} */ /* End of Doxygen Group NMSIS_Core_Bitmanip_Intrinsic */

#if defined(__INC_INTRINSIC_API) && (__INC_INTRINSIC_API == 1)
// deleted in gcc13, you can directly use b extension intrinisc api
//#include <rvintrin.h>
#endif

#endif /* defined(__BITMANIP_PRESENT) && (__BITMANIP_PRESENT == 1) */

#ifdef __cplusplus
}
#endif

#endif /* __CORE_FEATURE_BITMANIP__ */
