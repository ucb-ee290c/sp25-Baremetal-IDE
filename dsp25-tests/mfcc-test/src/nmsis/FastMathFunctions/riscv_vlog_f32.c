#include "dsp/fast_math_functions.h"
#include "riscv_common_tables.h"

#ifndef MFCC_VLOG_VEC_APPROX
#define MFCC_VLOG_VEC_APPROX 0
#endif


/**
  @ingroup groupFastMath
 */


/**
  @defgroup vlog Vector Log

  Compute the log values of a vector of samples.

 */

/**
  @addtogroup vlog
  @{
 */


RISCV_DSP_ATTRIBUTE void riscv_vlog_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize)
{
#if defined(RISCV_MATH_VECTOR) && (MFCC_VLOG_VEC_APPROX == 1)
   uint32_t blkCnt = blockSize;
   size_t l;

   while ((l = __riscv_vsetvl_e32m8(blkCnt)) > 0)
   {
      vfloat32m8_t vx = __riscv_vle32_v_f32m8(pSrc, l);
      vfloat32m8_t vone = __riscv_vfmv_v_f_f32m8(1.0f, l);

      /* Bring values closer to 1.0 to stabilize a short odd-power series. */
      vx = __riscv_vfsqrt_v_f32m8(vx, l);
      vx = __riscv_vfsqrt_v_f32m8(vx, l);
      vx = __riscv_vfsqrt_v_f32m8(vx, l);

      vfloat32m8_t vzNum = __riscv_vfsub_vv_f32m8(vx, vone, l);
      vfloat32m8_t vzDen = __riscv_vfadd_vv_f32m8(vx, vone, l);
      vfloat32m8_t vz = __riscv_vfdiv_vv_f32m8(vzNum, vzDen, l);

      {
         vfloat32m8_t vz2 = __riscv_vfmul_vv_f32m8(vz, vz, l);
         vfloat32m8_t vpoly = __riscv_vfmv_v_f_f32m8(0.1111111111f, l); /* 1/9 */

         /* Horner form for: z + z^3/3 + z^5/5 + z^7/7 + z^9/9 */
         vpoly = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(vpoly, vz2, l), 0.1428571429f, l); /* +1/7 */
         vpoly = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(vpoly, vz2, l), 0.2f, l);          /* +1/5 */
         vpoly = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(vpoly, vz2, l), 0.3333333333f, l); /* +1/3 */
         vpoly = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(vpoly, vz2, l), 1.0f, l);
         vpoly = __riscv_vfmul_vv_f32m8(vpoly, vz, l);

         /* ln(x) = 2^3 * 2 * ln(x^(1/8)) = 16 * series */
         vfloat32m8_t vln = __riscv_vfmul_vf_f32m8(vpoly, 16.0f, l);
         __riscv_vse32_v_f32m8(pDst, vln, l);
      }

      pSrc += l;
      pDst += l;
      blkCnt -= (uint32_t)l;
   }
#else
   uint32_t blkCnt; 

   blkCnt = blockSize;

   while (blkCnt > 0U)
   {
      /* C = log(A) */
  
      /* Calculate log and store result in destination buffer. */
      *pDst++ = logf(*pSrc++);
  
      /* Decrement loop counter */
      blkCnt--;
   }
#endif
}

/**
  @} end of vlog group
 */
