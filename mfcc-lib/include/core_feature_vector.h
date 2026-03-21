#ifndef __CORE_FEATURE_VECTOR__
#define __CORE_FEATURE_VECTOR__

/*
 * Vector Feature Configuration Macro:
 * 1. __VECTOR_PRESENT:  Define whether Vector Unit is present or not
 *   * 0: Not present
 *   * 1: Present
 */
#ifdef __cplusplus
 extern "C" {
#endif

#include "core_feature_base.h"

#if defined(__VECTOR_PRESENT) && (__VECTOR_PRESENT == 1)

/* ###########################  CPU Vector Intrinsic Functions ########################### */
/**
 * \defgroup NMSIS_Core_Vector_Intrinsic   Intrinsic Functions for Vector Instructions
 * \ingroup  NMSIS_Core
 * \brief    Functions that generate RISC-V Vector instructions.
 * \details
 *
 * RISC-V Vector Intrinsic APIs are provided directly through compiler generated intrinsic function.
 *
 * This intrinsic function support by compiler:
 *
 * For Nuclei RISC-V GCC 10.2, it is an very old and not ratified version(no longer supported).
 *
 * - API header file can be found in lib/gcc/riscv-nuclei-elf/<gcc_ver>/include/riscv_vector.h
 *
 * For Nuclei RISC-V GCC 13/Clang 17, the intrinsic API supported is v0.12 version, see
 * https://github.com/riscv-non-isa/rvv-intrinsic-doc/releases/tag/v0.12.0
 *
 * For Nuclei RISC-V GCC 14.2/Clang 19, the intrinsic API supported is v0.11.x version, see
 * https://github.com/riscv-non-isa/rvv-intrinsic-doc/tree/v0.11.x
 *   @{
 */

#if defined(__INC_INTRINSIC_API) && (__INC_INTRINSIC_API == 1)
#include <riscv_vector.h>
#endif

/**
 * \brief   Enable Vector Unit
 * \details
 * Set vector context status bits to enable vector unit,
 * and set state to initial
 */
__STATIC_FORCEINLINE void __enable_vector(void)
{
    __RV_CSR_CLEAR(CSR_MSTATUS, MSTATUS_VS);
    __RV_CSR_SET(CSR_MSTATUS, MSTATUS_VS_INITIAL);
}

/**
 * \brief   Disable Vector Unit
 * \details
 * Clear vector context status bits to disable vector unit
 */
__STATIC_FORCEINLINE void __disable_vector(void)
{
    __RV_CSR_CLEAR(CSR_MSTATUS, MSTATUS_VS);
}

/** @} */ /* End of Doxygen Group NMSIS_Core_Vector_Intrinsic */
#endif /* defined(__VECTOR_PRESENT) && (__VECTOR_PRESENT == 1) */

#ifdef __cplusplus
}
#endif

#endif /* __CORE_FEATURE_VECTOR__ */
