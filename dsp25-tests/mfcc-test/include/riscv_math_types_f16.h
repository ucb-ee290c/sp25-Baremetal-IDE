#ifndef RISCV_MATH_TYPES_F16_H
#define RISCV_MATH_TYPES_F16_H

//#include "riscv_math_types.h"

#ifdef   __cplusplus
extern "C"
{
#endif



/*

Check if the type __fp16 is available.
If it is not available, f16 version of the kernels
won't be built.

*/

#if (defined (__riscv_zfh))
  #undef RISCV_FLOAT16_SUPPORTED
  #define RISCV_FLOAT16_SUPPORTED 1
#endif




#if defined(RISCV_FLOAT16_SUPPORTED)

typedef _Float16 float16_t;

#define F16INFINITY ((float16_t)__builtin_inf())


#define F16_MAX   ((float16_t)__FLT16_MAX__)
#define F16_MIN   (-(float16_t)__FLT16_MAX__)

#define F16_ABSMAX   ((float16_t)__FLT16_MAX__)
#define F16_ABSMIN   ((float16_t)0.0f16)

#endif /* RISCV_FLOAT16_SUPPORTED*/

#ifdef   __cplusplus
}
#endif

#endif /* _RISCV_MATH_F16_H */
