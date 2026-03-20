#include "dsp/fast_math_functions.h"
#include "riscv_common_tables.h"


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
}

/**
  @} end of vlog group
 */
