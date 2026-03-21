#include "dsp/fast_math_functions.h"
#include "riscv_common_tables.h"

#ifndef MFCC_VLOG_VEC_APPROX
#define MFCC_VLOG_VEC_APPROX 0
#endif

__STATIC_FORCEINLINE float32_t mfcc_log_approx_f32_scalar(float32_t x)
{
   union
   {
      float32_t f;
      uint32_t u;
   } cvt;

   if (x <= 0.0f)
   {
      return -FLT_MAX;
   }

   cvt.f = x;
   {
      uint32_t expRaw = (cvt.u >> 23) & 0xFFU;

      if (expRaw == 0xFFU)
      {
         return x;
      }

      if (expRaw == 0U)
      {
         return logf(x);
      }

      {
         int32_t e = (int32_t)expRaw - 126;
         float32_t m;
         float32_t y;
         float32_t y2;
         float32_t poly;

         cvt.u = (cvt.u & 0x007FFFFFU) | 0x3F000000U;
         m = cvt.f;

         if (m < 0.7071067811865476f)
         {
            m *= 2.0f;
            e -= 1;
         }

         y = (m - 1.0f) / (m + 1.0f);
         y2 = y * y;

         poly = 0.1111111111111111f;
         poly = (poly * y2) + 0.1428571428571429f;
         poly = (poly * y2) + 0.2f;
         poly = (poly * y2) + 0.3333333333333333f;
         poly = (poly * y2) + 1.0f;

         return (2.0f * y * poly) + ((float32_t)e * 0.6931471805599453f);
      }
   }
}


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
#if (MFCC_VLOG_VEC_APPROX == 1)
   uint32_t blkCnt = blockSize;

   while (blkCnt > 0U)
   {
      *pDst++ = mfcc_log_approx_f32_scalar(*pSrc++);
      blkCnt--;
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
