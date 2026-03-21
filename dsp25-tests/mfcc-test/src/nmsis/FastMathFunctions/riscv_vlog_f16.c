#include "dsp/fast_math_functions_f16.h"
#include "dsp/support_functions_f16.h"

#if defined(RISCV_FLOAT16_SUPPORTED)

#ifndef MFCC_VLOG_VEC_APPROX
#define MFCC_VLOG_VEC_APPROX 0
#endif

/* Degree of the polynomial approximation */
#define NB_DEG_LOGF16 3

/*
Related to the Log2 of the number of approximations.
For instance, with 3 there are 1 + 2^3 polynomials
*/
#define NB_DIV_LOGF16 3

/* Length of the LUT table */
#define NB_LUT_LOGF16 (NB_DEG_LOGF16+1)*(1 + (1<<NB_DIV_LOGF16))


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
#if defined(RISCV_MATH_VECTOR_F16) && (MFCC_VLOG_VEC_APPROX == 1)
   uint32_t blkCnt = blockSize;
   size_t l;

   while ((l = __riscv_vsetvl_e16m8(blkCnt)) > 0)
   {
      vfloat16m8_t vx = __riscv_vle16_v_f16m8(pSrc, l);
      vfloat16m8_t vone = __riscv_vfmv_v_f_f16m8(1.0f16, l);

      /* Bring values closer to 1.0 to stabilize a short odd-power series. */
      vx = __riscv_vfsqrt_v_f16m8(vx, l);
      vx = __riscv_vfsqrt_v_f16m8(vx, l);
      vx = __riscv_vfsqrt_v_f16m8(vx, l);

      vfloat16m8_t vzNum = __riscv_vfsub_vv_f16m8(vx, vone, l);
      vfloat16m8_t vzDen = __riscv_vfadd_vv_f16m8(vx, vone, l);
      vfloat16m8_t vz = __riscv_vfdiv_vv_f16m8(vzNum, vzDen, l);

      {
         vfloat16m8_t vz2 = __riscv_vfmul_vv_f16m8(vz, vz, l);
         vfloat16m8_t vpoly = __riscv_vfmv_v_f_f16m8(0.1111f16, l); /* 1/9 */

         /* Horner form for: z + z^3/3 + z^5/5 + z^7/7 + z^9/9 */
         vpoly = __riscv_vfadd_vf_f16m8(__riscv_vfmul_vv_f16m8(vpoly, vz2, l), 0.1428f16, l); /* +1/7 */
         vpoly = __riscv_vfadd_vf_f16m8(__riscv_vfmul_vv_f16m8(vpoly, vz2, l), 0.2f16, l);    /* +1/5 */
         vpoly = __riscv_vfadd_vf_f16m8(__riscv_vfmul_vv_f16m8(vpoly, vz2, l), 0.3333f16, l); /* +1/3 */
         vpoly = __riscv_vfadd_vf_f16m8(__riscv_vfmul_vv_f16m8(vpoly, vz2, l), 1.0f16, l);
         vpoly = __riscv_vfmul_vv_f16m8(vpoly, vz, l);

         /* ln(x) = 2^3 * 2 * ln(x^(1/8)) = 16 * series */
         vfloat16m8_t vln = __riscv_vfmul_vf_f16m8(vpoly, 16.0f16, l);
         __riscv_vse16_v_f16m8(pDst, vln, l);
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
