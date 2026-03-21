#include "dsp/fast_math_functions_f16.h"
#include "dsp/support_functions_f16.h"

#if defined(RISCV_FLOAT16_SUPPORTED)

#ifndef MFCC_VLOG_VEC_APPROX
#define MFCC_VLOG_VEC_APPROX 0
#endif

#if (MFCC_VLOG_VEC_APPROX == 0)
  /* Degree of the polynomial approximation */
  #define NB_DEG_LOGF16 3

  /*
  Related to the Log2 of the number of approximations.
  For instance, with 3 there are 1 + 2^3 polynomials
  */
  #define NB_DIV_LOGF16 3

  /* Length of the LUT table */
  #define NB_LUT_LOGF16 (NB_DEG_LOGF16+1)*(1 + (1<<NB_DIV_LOGF16))
#endif


/*

LUT of polynomial approximations.

Could be generated with:

ClearAll[lut, coefs, nb, deg];
nb = 3;
deg = 3;
lut = Table[
   MiniMaxApproximation[
     Log[x/2^nb + i], {x, {10^-6, 1.0/2^nb}, deg, 0},
     MaxIterations -> 1000][[2, 1]], {i, 1, 2, (1.0/2^nb)}];
coefs = Chop@Flatten[CoefficientList[lut, x]];

*/
#if (MFCC_VLOG_VEC_APPROX == 0)
static float16_t lut_logf16[NB_LUT_LOGF16]={
   0,0.125,-0.00781197,0.00063974,0.117783,
   0.111111,-0.00617212,0.000447935,0.223144,
   0.1,-0.00499952,0.000327193,0.318454,0.0909091,
   -0.00413191,0.000246234,0.405465,0.0833333,
   -0.00347199,0.000189928,0.485508,0.0769231,
   -0.00295841,0.00014956,0.559616,0.0714286,
   -0.0025509,0.000119868,0.628609,0.0666667,
   -0.00222213,0.0000975436,0.693147,
   0.0625,-0.00195305,0.0000804357};


static float16_t logf16_scalar(float16_t x)
{
    int16_t i =  riscv_typecast_s16_f16(x);

    int32_t vecExpUnBiased = (i >> 10) - 15;
    i = i - (vecExpUnBiased << 10);
    float16_t vecTmpFlt1 = riscv_typecast_f16_s16(i);

    float16_t *lut;
    int n;
    float16_t tmp,v;

    tmp = ((_Float16)vecTmpFlt1 - 1.0f16) * (1 << NB_DIV_LOGF16);
    n = (int)tmp;
    if (n < 0)
    {
      n = 0;
    }
    else if (n > (1 << NB_DIV_LOGF16))
    {
      n = (1 << NB_DIV_LOGF16);
    }
    v = (_Float16)tmp - (_Float16)n;

    lut = lut_logf16 + n * (1+NB_DEG_LOGF16);

    float16_t res = lut[NB_DEG_LOGF16-1];
    for(int j=NB_DEG_LOGF16-2; j >=0 ; j--)
    {
       res = (_Float16)lut[j] + (_Float16)v * (_Float16)res;
    }

    res = (_Float16)res + 0.693147f16 * (_Float16)vecExpUnBiased;


    return(res);
}
#endif

static __STATIC_FORCEINLINE float16_t mfcc_log_approx_f16_scalar(float16_t x)
{
    uint16_t bits = (uint16_t)riscv_typecast_s16_f16(x);
    uint16_t expRaw = (bits >> 10) & 0x1FU;
    int32_t e;
    uint16_t mBits;
    float32_t m;
    float32_t y;
    float32_t y2;
    float32_t poly;
    float32_t ln;

    if ((_Float16)x <= 0.0f16)
    {
      return -F16_MAX;
    }

    if (expRaw == 0x1FU)
    {
      return x;
    }

    if (expRaw == 0U)
    {
      return (float16_t)logf((float32_t)x);
    }

    e = (int32_t)expRaw - 14;
    mBits = (bits & 0x03FFU) | (14U << 10);
    m = (float32_t)riscv_typecast_f16_s16((int16_t)mBits);

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

    ln = (2.0f * y * poly) + ((float32_t)e * 0.6931471805599453f);
    return (float16_t)ln;
}


/**
  @ingroup groupFastMath
 */

/**
  @addtogroup vlog
  @{
 */

/**
  @brief         Floating-point vector of log values.
  @param[in]     pSrc       points to the input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
 */


RISCV_DSP_ATTRIBUTE void riscv_vlog_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t blockSize)
{
#if (MFCC_VLOG_VEC_APPROX == 1)
   uint32_t blkCnt = blockSize;

   while (blkCnt > 0U)
   {
      *pDst++ = mfcc_log_approx_f16_scalar(*pSrc++);
      blkCnt--;
   }
#else
   uint32_t blkCnt;

   blkCnt = blockSize;

   while (blkCnt > 0U)
   {
      /* C = log(A) */

      /* Calculate log and store result in destination buffer. */
      *pDst++ = logf16_scalar(*pSrc++);

      /* Decrement loop counter */
      blkCnt--;
   }
#endif
}



/**
  @} end of vlog group
 */


#endif /* #if defined(RISCV_FLOAT16_SUPPORTED) */
