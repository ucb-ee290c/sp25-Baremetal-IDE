#include "dsp/transform_functions.h"


/* ----------------------------------------------------------------------
 * Internal helper function used by the FFTs
 * -------------------------------------------------------------------- */

void riscv_radix8_butterfly_f32(
  float32_t * pSrc,
  uint16_t fftLen,
  const float32_t * pCoef,
  uint16_t twidCoefModifier);

/**
  brief         Core function for the floating-point CFFT butterfly process.
  param[in,out] pSrc             points to the in-place buffer of floating-point data type.
  param[in]     fftLen           length of the FFT.
  param[in]     pCoef            points to the twiddle coefficient buffer.
  param[in]     twidCoefModifier twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table.
  return        none
*/

RISCV_DSP_ATTRIBUTE void riscv_radix8_butterfly_f32(
  float32_t * pSrc,
  uint16_t fftLen,
  const float32_t * pCoef,
  uint16_t twidCoefModifier)
{
   uint32_t ia1, ia2, ia3, ia4, ia5, ia6, ia7;
   uint32_t i1, i2, i3, i4, i5, i6, i7, i8;
   uint32_t id;
   uint32_t n1, n2, j;

   float32_t r1, r2, r3, r4, r5, r6, r7, r8;
   float32_t t1, t2;
   float32_t s1, s2, s3, s4, s5, s6, s7, s8;
   float32_t p1, p2, p3, p4;
   float32_t co2, co3, co4, co5, co6, co7, co8;
   float32_t si2, si3, si4, si5, si6, si7, si8;
   const float32_t C81 = 0.70710678118f;

   n2 = fftLen;

   do
   {
      n1 = n2;
      n2 = n2 >> 3;
      i1 = 0;

#if defined(RISCV_MATH_VECTOR)
      {
         uint32_t blkCnt = fftLen / n1;
         size_t vl;
         ptrdiff_t stride = (ptrdiff_t)(2U * n1 * sizeof(float32_t));
         float32_t *pI1 = pSrc;
         float32_t *pI2 = pSrc + (2U * n2);
         float32_t *pI3 = pSrc + (4U * n2);
         float32_t *pI4 = pSrc + (6U * n2);
         float32_t *pI5 = pSrc + (8U * n2);
         float32_t *pI6 = pSrc + (10U * n2);
         float32_t *pI7 = pSrc + (12U * n2);
         float32_t *pI8 = pSrc + (14U * n2);

         while ((vl = __riscv_vsetvl_e32m8(blkCnt)) > 0)
         {
            vfloat32m8_t v1r = __riscv_vlse32_v_f32m8(pI1, stride, vl);
            vfloat32m8_t v1i = __riscv_vlse32_v_f32m8(pI1 + 1, stride, vl);
            vfloat32m8_t v2r = __riscv_vlse32_v_f32m8(pI2, stride, vl);
            vfloat32m8_t v2i = __riscv_vlse32_v_f32m8(pI2 + 1, stride, vl);
            vfloat32m8_t v3r = __riscv_vlse32_v_f32m8(pI3, stride, vl);
            vfloat32m8_t v3i = __riscv_vlse32_v_f32m8(pI3 + 1, stride, vl);
            vfloat32m8_t v4r = __riscv_vlse32_v_f32m8(pI4, stride, vl);
            vfloat32m8_t v4i = __riscv_vlse32_v_f32m8(pI4 + 1, stride, vl);
            vfloat32m8_t v5r = __riscv_vlse32_v_f32m8(pI5, stride, vl);
            vfloat32m8_t v5i = __riscv_vlse32_v_f32m8(pI5 + 1, stride, vl);
            vfloat32m8_t v6r = __riscv_vlse32_v_f32m8(pI6, stride, vl);
            vfloat32m8_t v6i = __riscv_vlse32_v_f32m8(pI6 + 1, stride, vl);
            vfloat32m8_t v7r = __riscv_vlse32_v_f32m8(pI7, stride, vl);
            vfloat32m8_t v7i = __riscv_vlse32_v_f32m8(pI7 + 1, stride, vl);
            vfloat32m8_t v8r = __riscv_vlse32_v_f32m8(pI8, stride, vl);
            vfloat32m8_t v8i = __riscv_vlse32_v_f32m8(pI8 + 1, stride, vl);

            vfloat32m8_t vr1 = __riscv_vfadd_vv_f32m8(v1r, v5r, vl);
            vfloat32m8_t vr5 = __riscv_vfsub_vv_f32m8(v1r, v5r, vl);
            vfloat32m8_t vr2 = __riscv_vfadd_vv_f32m8(v2r, v6r, vl);
            vfloat32m8_t vr6 = __riscv_vfsub_vv_f32m8(v2r, v6r, vl);
            vfloat32m8_t vr3 = __riscv_vfadd_vv_f32m8(v3r, v7r, vl);
            vfloat32m8_t vr7 = __riscv_vfsub_vv_f32m8(v3r, v7r, vl);
            vfloat32m8_t vr4 = __riscv_vfadd_vv_f32m8(v4r, v8r, vl);
            vfloat32m8_t vr8 = __riscv_vfsub_vv_f32m8(v4r, v8r, vl);

            vfloat32m8_t vt1 = __riscv_vfsub_vv_f32m8(vr1, vr3, vl);
            vr1 = __riscv_vfadd_vv_f32m8(vr1, vr3, vl);
            vr3 = __riscv_vfsub_vv_f32m8(vr2, vr4, vl);
            vr2 = __riscv_vfadd_vv_f32m8(vr2, vr4, vl);

            vfloat32m8_t vOut1r = __riscv_vfadd_vv_f32m8(vr1, vr2, vl);
            vfloat32m8_t vOut5r = __riscv_vfsub_vv_f32m8(vr1, vr2, vl);

            vfloat32m8_t vs1 = __riscv_vfadd_vv_f32m8(v1i, v5i, vl);
            vfloat32m8_t vs5 = __riscv_vfsub_vv_f32m8(v1i, v5i, vl);
            vfloat32m8_t vs2 = __riscv_vfadd_vv_f32m8(v2i, v6i, vl);
            vfloat32m8_t vs6 = __riscv_vfsub_vv_f32m8(v2i, v6i, vl);
            vfloat32m8_t vs3 = __riscv_vfadd_vv_f32m8(v3i, v7i, vl);
            vfloat32m8_t vs7 = __riscv_vfsub_vv_f32m8(v3i, v7i, vl);
            vfloat32m8_t vs4 = __riscv_vfadd_vv_f32m8(v4i, v8i, vl);
            vfloat32m8_t vs8 = __riscv_vfsub_vv_f32m8(v4i, v8i, vl);

            vfloat32m8_t vt2 = __riscv_vfsub_vv_f32m8(vs1, vs3, vl);
            vs1 = __riscv_vfadd_vv_f32m8(vs1, vs3, vl);
            vs3 = __riscv_vfsub_vv_f32m8(vs2, vs4, vl);
            vs2 = __riscv_vfadd_vv_f32m8(vs2, vs4, vl);

            vfloat32m8_t vOut1i = __riscv_vfadd_vv_f32m8(vs1, vs2, vl);
            vfloat32m8_t vOut5i = __riscv_vfsub_vv_f32m8(vs1, vs2, vl);
            vfloat32m8_t vOut3r = __riscv_vfadd_vv_f32m8(vt1, vs3, vl);
            vfloat32m8_t vOut7r = __riscv_vfsub_vv_f32m8(vt1, vs3, vl);
            vfloat32m8_t vOut3i = __riscv_vfsub_vv_f32m8(vt2, vr3, vl);
            vfloat32m8_t vOut7i = __riscv_vfadd_vv_f32m8(vt2, vr3, vl);

            vfloat32m8_t vTmpR1 = __riscv_vfmul_vf_f32m8(__riscv_vfsub_vv_f32m8(vr6, vr8, vl), C81, vl);
            vr6 = __riscv_vfmul_vf_f32m8(__riscv_vfadd_vv_f32m8(vr6, vr8, vl), C81, vl);
            vfloat32m8_t vTmpI1 = __riscv_vfmul_vf_f32m8(__riscv_vfsub_vv_f32m8(vs6, vs8, vl), C81, vl);
            vs6 = __riscv_vfmul_vf_f32m8(__riscv_vfadd_vv_f32m8(vs6, vs8, vl), C81, vl);

            vfloat32m8_t vTmpR2 = __riscv_vfsub_vv_f32m8(vr5, vTmpR1, vl);
            vr5 = __riscv_vfadd_vv_f32m8(vr5, vTmpR1, vl);
            vr8 = __riscv_vfsub_vv_f32m8(vr7, vr6, vl);
            vr7 = __riscv_vfadd_vv_f32m8(vr7, vr6, vl);
            vfloat32m8_t vTmpI2 = __riscv_vfsub_vv_f32m8(vs5, vTmpI1, vl);
            vs5 = __riscv_vfadd_vv_f32m8(vs5, vTmpI1, vl);
            vs8 = __riscv_vfsub_vv_f32m8(vs7, vs6, vl);
            vs7 = __riscv_vfadd_vv_f32m8(vs7, vs6, vl);

            vfloat32m8_t vOut2r = __riscv_vfadd_vv_f32m8(vr5, vs7, vl);
            vfloat32m8_t vOut8r = __riscv_vfsub_vv_f32m8(vr5, vs7, vl);
            vfloat32m8_t vOut6r = __riscv_vfadd_vv_f32m8(vTmpR2, vs8, vl);
            vfloat32m8_t vOut4r = __riscv_vfsub_vv_f32m8(vTmpR2, vs8, vl);
            vfloat32m8_t vOut2i = __riscv_vfsub_vv_f32m8(vs5, vr7, vl);
            vfloat32m8_t vOut8i = __riscv_vfadd_vv_f32m8(vs5, vr7, vl);
            vfloat32m8_t vOut6i = __riscv_vfsub_vv_f32m8(vTmpI2, vr8, vl);
            vfloat32m8_t vOut4i = __riscv_vfadd_vv_f32m8(vTmpI2, vr8, vl);

            __riscv_vsse32_v_f32m8(pI1, stride, vOut1r, vl);
            __riscv_vsse32_v_f32m8(pI1 + 1, stride, vOut1i, vl);
            __riscv_vsse32_v_f32m8(pI2, stride, vOut2r, vl);
            __riscv_vsse32_v_f32m8(pI2 + 1, stride, vOut2i, vl);
            __riscv_vsse32_v_f32m8(pI3, stride, vOut3r, vl);
            __riscv_vsse32_v_f32m8(pI3 + 1, stride, vOut3i, vl);
            __riscv_vsse32_v_f32m8(pI4, stride, vOut4r, vl);
            __riscv_vsse32_v_f32m8(pI4 + 1, stride, vOut4i, vl);
            __riscv_vsse32_v_f32m8(pI5, stride, vOut5r, vl);
            __riscv_vsse32_v_f32m8(pI5 + 1, stride, vOut5i, vl);
            __riscv_vsse32_v_f32m8(pI6, stride, vOut6r, vl);
            __riscv_vsse32_v_f32m8(pI6 + 1, stride, vOut6i, vl);
            __riscv_vsse32_v_f32m8(pI7, stride, vOut7r, vl);
            __riscv_vsse32_v_f32m8(pI7 + 1, stride, vOut7i, vl);
            __riscv_vsse32_v_f32m8(pI8, stride, vOut8r, vl);
            __riscv_vsse32_v_f32m8(pI8 + 1, stride, vOut8i, vl);

            {
               uint32_t elemInc = (uint32_t)(2U * n1 * (uint32_t)vl);
               pI1 += elemInc;
               pI2 += elemInc;
               pI3 += elemInc;
               pI4 += elemInc;
               pI5 += elemInc;
               pI6 += elemInc;
               pI7 += elemInc;
               pI8 += elemInc;
            }
            blkCnt -= (uint32_t)vl;
         }
      }
#else
      do
      {
         i2 = i1 + n2;
         i3 = i2 + n2;
         i4 = i3 + n2;
         i5 = i4 + n2;
         i6 = i5 + n2;
         i7 = i6 + n2;
         i8 = i7 + n2;
         r1 = pSrc[2 * i1] + pSrc[2 * i5];
         r5 = pSrc[2 * i1] - pSrc[2 * i5];
         r2 = pSrc[2 * i2] + pSrc[2 * i6];
         r6 = pSrc[2 * i2] - pSrc[2 * i6];
         r3 = pSrc[2 * i3] + pSrc[2 * i7];
         r7 = pSrc[2 * i3] - pSrc[2 * i7];
         r4 = pSrc[2 * i4] + pSrc[2 * i8];
         r8 = pSrc[2 * i4] - pSrc[2 * i8];
         t1 = r1 - r3;
         r1 = r1 + r3;
         r3 = r2 - r4;
         r2 = r2 + r4;
         pSrc[2 * i1] = r1 + r2;
         pSrc[2 * i5] = r1 - r2;
         r1 = pSrc[2 * i1 + 1] + pSrc[2 * i5 + 1];
         s5 = pSrc[2 * i1 + 1] - pSrc[2 * i5 + 1];
         r2 = pSrc[2 * i2 + 1] + pSrc[2 * i6 + 1];
         s6 = pSrc[2 * i2 + 1] - pSrc[2 * i6 + 1];
         s3 = pSrc[2 * i3 + 1] + pSrc[2 * i7 + 1];
         s7 = pSrc[2 * i3 + 1] - pSrc[2 * i7 + 1];
         r4 = pSrc[2 * i4 + 1] + pSrc[2 * i8 + 1];
         s8 = pSrc[2 * i4 + 1] - pSrc[2 * i8 + 1];
         t2 = r1 - s3;
         r1 = r1 + s3;
         s3 = r2 - r4;
         r2 = r2 + r4;
         pSrc[2 * i1 + 1] = r1 + r2;
         pSrc[2 * i5 + 1] = r1 - r2;
         pSrc[2 * i3]     = t1 + s3;
         pSrc[2 * i7]     = t1 - s3;
         pSrc[2 * i3 + 1] = t2 - r3;
         pSrc[2 * i7 + 1] = t2 + r3;
         r1 = (r6 - r8) * C81;
         r6 = (r6 + r8) * C81;
         r2 = (s6 - s8) * C81;
         s6 = (s6 + s8) * C81;
         t1 = r5 - r1;
         r5 = r5 + r1;
         r8 = r7 - r6;
         r7 = r7 + r6;
         t2 = s5 - r2;
         s5 = s5 + r2;
         s8 = s7 - s6;
         s7 = s7 + s6;
         pSrc[2 * i2]     = r5 + s7;
         pSrc[2 * i8]     = r5 - s7;
         pSrc[2 * i6]     = t1 + s8;
         pSrc[2 * i4]     = t1 - s8;
         pSrc[2 * i2 + 1] = s5 - r7;
         pSrc[2 * i8 + 1] = s5 + r7;
         pSrc[2 * i6 + 1] = t2 - r8;
         pSrc[2 * i4 + 1] = t2 + r8;

         i1 += n1;
      } while (i1 < fftLen);
#endif

      if (n2 < 8)
         break;

      ia1 = 0;
      j = 1;

      do
      {
         /*  index calculation for the coefficients */
         id  = ia1 + twidCoefModifier;
         ia1 = id;
         ia2 = ia1 + id;
         ia3 = ia2 + id;
         ia4 = ia3 + id;
         ia5 = ia4 + id;
         ia6 = ia5 + id;
         ia7 = ia6 + id;

         co2 = pCoef[2 * ia1];
         co3 = pCoef[2 * ia2];
         co4 = pCoef[2 * ia3];
         co5 = pCoef[2 * ia4];
         co6 = pCoef[2 * ia5];
         co7 = pCoef[2 * ia6];
         co8 = pCoef[2 * ia7];
         si2 = pCoef[2 * ia1 + 1];
         si3 = pCoef[2 * ia2 + 1];
         si4 = pCoef[2 * ia3 + 1];
         si5 = pCoef[2 * ia4 + 1];
         si6 = pCoef[2 * ia5 + 1];
         si7 = pCoef[2 * ia6 + 1];
         si8 = pCoef[2 * ia7 + 1];

         i1 = j;

#if defined(RISCV_MATH_VECTOR)
         {
            uint32_t blkCnt = (fftLen - j + n1 - 1U) / n1;
            size_t vl;
            ptrdiff_t stride = (ptrdiff_t)(2U * n1 * sizeof(float32_t));
            float32_t *pI1 = pSrc + (2U * j);
            float32_t *pI2 = pI1 + (2U * n2);
            float32_t *pI3 = pI2 + (2U * n2);
            float32_t *pI4 = pI3 + (2U * n2);
            float32_t *pI5 = pI4 + (2U * n2);
            float32_t *pI6 = pI5 + (2U * n2);
            float32_t *pI7 = pI6 + (2U * n2);
            float32_t *pI8 = pI7 + (2U * n2);

            while ((vl = __riscv_vsetvl_e32m8(blkCnt)) > 0)
            {
               vfloat32m8_t v1r = __riscv_vlse32_v_f32m8(pI1, stride, vl);
               vfloat32m8_t v1i = __riscv_vlse32_v_f32m8(pI1 + 1, stride, vl);
               vfloat32m8_t v2r = __riscv_vlse32_v_f32m8(pI2, stride, vl);
               vfloat32m8_t v2i = __riscv_vlse32_v_f32m8(pI2 + 1, stride, vl);
               vfloat32m8_t v3r = __riscv_vlse32_v_f32m8(pI3, stride, vl);
               vfloat32m8_t v3i = __riscv_vlse32_v_f32m8(pI3 + 1, stride, vl);
               vfloat32m8_t v4r = __riscv_vlse32_v_f32m8(pI4, stride, vl);
               vfloat32m8_t v4i = __riscv_vlse32_v_f32m8(pI4 + 1, stride, vl);
               vfloat32m8_t v5r = __riscv_vlse32_v_f32m8(pI5, stride, vl);
               vfloat32m8_t v5i = __riscv_vlse32_v_f32m8(pI5 + 1, stride, vl);
               vfloat32m8_t v6r = __riscv_vlse32_v_f32m8(pI6, stride, vl);
               vfloat32m8_t v6i = __riscv_vlse32_v_f32m8(pI6 + 1, stride, vl);
               vfloat32m8_t v7r = __riscv_vlse32_v_f32m8(pI7, stride, vl);
               vfloat32m8_t v7i = __riscv_vlse32_v_f32m8(pI7 + 1, stride, vl);
               vfloat32m8_t v8r = __riscv_vlse32_v_f32m8(pI8, stride, vl);
               vfloat32m8_t v8i = __riscv_vlse32_v_f32m8(pI8 + 1, stride, vl);

               vfloat32m8_t vr1 = __riscv_vfadd_vv_f32m8(v1r, v5r, vl);
               vfloat32m8_t vr5 = __riscv_vfsub_vv_f32m8(v1r, v5r, vl);
               vfloat32m8_t vr2 = __riscv_vfadd_vv_f32m8(v2r, v6r, vl);
               vfloat32m8_t vr6 = __riscv_vfsub_vv_f32m8(v2r, v6r, vl);
               vfloat32m8_t vr3 = __riscv_vfadd_vv_f32m8(v3r, v7r, vl);
               vfloat32m8_t vr7 = __riscv_vfsub_vv_f32m8(v3r, v7r, vl);
               vfloat32m8_t vr4 = __riscv_vfadd_vv_f32m8(v4r, v8r, vl);
               vfloat32m8_t vr8 = __riscv_vfsub_vv_f32m8(v4r, v8r, vl);

               vfloat32m8_t vt1 = __riscv_vfsub_vv_f32m8(vr1, vr3, vl);
               vr1 = __riscv_vfadd_vv_f32m8(vr1, vr3, vl);
               vr3 = __riscv_vfsub_vv_f32m8(vr2, vr4, vl);
               vr2 = __riscv_vfadd_vv_f32m8(vr2, vr4, vl);

               vfloat32m8_t vOut1r = __riscv_vfadd_vv_f32m8(vr1, vr2, vl);
               vfloat32m8_t vR2 = __riscv_vfsub_vv_f32m8(vr1, vr2, vl);

               vfloat32m8_t vs1 = __riscv_vfadd_vv_f32m8(v1i, v5i, vl);
               vfloat32m8_t vs5 = __riscv_vfsub_vv_f32m8(v1i, v5i, vl);
               vfloat32m8_t vs2 = __riscv_vfadd_vv_f32m8(v2i, v6i, vl);
               vfloat32m8_t vs6 = __riscv_vfsub_vv_f32m8(v2i, v6i, vl);
               vfloat32m8_t vs3 = __riscv_vfadd_vv_f32m8(v3i, v7i, vl);
               vfloat32m8_t vs7 = __riscv_vfsub_vv_f32m8(v3i, v7i, vl);
               vfloat32m8_t vs4 = __riscv_vfadd_vv_f32m8(v4i, v8i, vl);
               vfloat32m8_t vs8 = __riscv_vfsub_vv_f32m8(v4i, v8i, vl);

               vfloat32m8_t vt2 = __riscv_vfsub_vv_f32m8(vs1, vs3, vl);
               vs1 = __riscv_vfadd_vv_f32m8(vs1, vs3, vl);
               vs3 = __riscv_vfsub_vv_f32m8(vs2, vs4, vl);
               vs2 = __riscv_vfadd_vv_f32m8(vs2, vs4, vl);

               vfloat32m8_t vR1 = __riscv_vfadd_vv_f32m8(vt1, vs3, vl);
               vfloat32m8_t vT1 = __riscv_vfsub_vv_f32m8(vt1, vs3, vl);
               vfloat32m8_t vOut1i = __riscv_vfadd_vv_f32m8(vs1, vs2, vl);
               vfloat32m8_t vS2 = __riscv_vfsub_vv_f32m8(vs1, vs2, vl);
               vfloat32m8_t vS1 = __riscv_vfsub_vv_f32m8(vt2, vr3, vl);
               vfloat32m8_t vT2 = __riscv_vfadd_vv_f32m8(vt2, vr3, vl);

               vfloat32m8_t vOut5r = __riscv_vfadd_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vR2, co5, vl),
                   __riscv_vfmul_vf_f32m8(vS2, si5, vl), vl);
               vfloat32m8_t vOut5i = __riscv_vfsub_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vS2, co5, vl),
                   __riscv_vfmul_vf_f32m8(vR2, si5, vl), vl);
               vfloat32m8_t vOut3r = __riscv_vfadd_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vR1, co3, vl),
                   __riscv_vfmul_vf_f32m8(vS1, si3, vl), vl);
               vfloat32m8_t vOut3i = __riscv_vfsub_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vS1, co3, vl),
                   __riscv_vfmul_vf_f32m8(vR1, si3, vl), vl);
               vfloat32m8_t vOut7r = __riscv_vfadd_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vT1, co7, vl),
                   __riscv_vfmul_vf_f32m8(vT2, si7, vl), vl);
               vfloat32m8_t vOut7i = __riscv_vfsub_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vT2, co7, vl),
                   __riscv_vfmul_vf_f32m8(vT1, si7, vl), vl);

               vfloat32m8_t vR1b = __riscv_vfmul_vf_f32m8(__riscv_vfsub_vv_f32m8(vr6, vr8, vl), C81, vl);
               vr6 = __riscv_vfmul_vf_f32m8(__riscv_vfadd_vv_f32m8(vr6, vr8, vl), C81, vl);
               vfloat32m8_t vS1b = __riscv_vfmul_vf_f32m8(__riscv_vfsub_vv_f32m8(vs6, vs8, vl), C81, vl);
               vs6 = __riscv_vfmul_vf_f32m8(__riscv_vfadd_vv_f32m8(vs6, vs8, vl), C81, vl);

               vfloat32m8_t vT1b = __riscv_vfsub_vv_f32m8(vr5, vR1b, vl);
               vr5 = __riscv_vfadd_vv_f32m8(vr5, vR1b, vl);
               vr8 = __riscv_vfsub_vv_f32m8(vr7, vr6, vl);
               vr7 = __riscv_vfadd_vv_f32m8(vr7, vr6, vl);
               vfloat32m8_t vT2b = __riscv_vfsub_vv_f32m8(vs5, vS1b, vl);
               vs5 = __riscv_vfadd_vv_f32m8(vs5, vS1b, vl);
               vs8 = __riscv_vfsub_vv_f32m8(vs7, vs6, vl);
               vs7 = __riscv_vfadd_vv_f32m8(vs7, vs6, vl);

               vfloat32m8_t vR1c = __riscv_vfadd_vv_f32m8(vr5, vs7, vl);
               vfloat32m8_t vR5c = __riscv_vfsub_vv_f32m8(vr5, vs7, vl);
               vfloat32m8_t vR6c = __riscv_vfadd_vv_f32m8(vT1b, vs8, vl);
               vfloat32m8_t vT1c = __riscv_vfsub_vv_f32m8(vT1b, vs8, vl);
               vfloat32m8_t vS1c = __riscv_vfsub_vv_f32m8(vs5, vr7, vl);
               vfloat32m8_t vS5c = __riscv_vfadd_vv_f32m8(vs5, vr7, vl);
               vfloat32m8_t vS6c = __riscv_vfsub_vv_f32m8(vT2b, vr8, vl);
               vfloat32m8_t vT2c = __riscv_vfadd_vv_f32m8(vT2b, vr8, vl);

               vfloat32m8_t vOut2r = __riscv_vfadd_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vR1c, co2, vl),
                   __riscv_vfmul_vf_f32m8(vS1c, si2, vl), vl);
               vfloat32m8_t vOut2i = __riscv_vfsub_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vS1c, co2, vl),
                   __riscv_vfmul_vf_f32m8(vR1c, si2, vl), vl);
               vfloat32m8_t vOut8r = __riscv_vfadd_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vR5c, co8, vl),
                   __riscv_vfmul_vf_f32m8(vS5c, si8, vl), vl);
               vfloat32m8_t vOut8i = __riscv_vfsub_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vS5c, co8, vl),
                   __riscv_vfmul_vf_f32m8(vR5c, si8, vl), vl);
               vfloat32m8_t vOut6r = __riscv_vfadd_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vR6c, co6, vl),
                   __riscv_vfmul_vf_f32m8(vS6c, si6, vl), vl);
               vfloat32m8_t vOut6i = __riscv_vfsub_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vS6c, co6, vl),
                   __riscv_vfmul_vf_f32m8(vR6c, si6, vl), vl);
               vfloat32m8_t vOut4r = __riscv_vfadd_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vT1c, co4, vl),
                   __riscv_vfmul_vf_f32m8(vT2c, si4, vl), vl);
               vfloat32m8_t vOut4i = __riscv_vfsub_vv_f32m8(
                   __riscv_vfmul_vf_f32m8(vT2c, co4, vl),
                   __riscv_vfmul_vf_f32m8(vT1c, si4, vl), vl);

               __riscv_vsse32_v_f32m8(pI1, stride, vOut1r, vl);
               __riscv_vsse32_v_f32m8(pI1 + 1, stride, vOut1i, vl);
               __riscv_vsse32_v_f32m8(pI2, stride, vOut2r, vl);
               __riscv_vsse32_v_f32m8(pI2 + 1, stride, vOut2i, vl);
               __riscv_vsse32_v_f32m8(pI3, stride, vOut3r, vl);
               __riscv_vsse32_v_f32m8(pI3 + 1, stride, vOut3i, vl);
               __riscv_vsse32_v_f32m8(pI4, stride, vOut4r, vl);
               __riscv_vsse32_v_f32m8(pI4 + 1, stride, vOut4i, vl);
               __riscv_vsse32_v_f32m8(pI5, stride, vOut5r, vl);
               __riscv_vsse32_v_f32m8(pI5 + 1, stride, vOut5i, vl);
               __riscv_vsse32_v_f32m8(pI6, stride, vOut6r, vl);
               __riscv_vsse32_v_f32m8(pI6 + 1, stride, vOut6i, vl);
               __riscv_vsse32_v_f32m8(pI7, stride, vOut7r, vl);
               __riscv_vsse32_v_f32m8(pI7 + 1, stride, vOut7i, vl);
               __riscv_vsse32_v_f32m8(pI8, stride, vOut8r, vl);
               __riscv_vsse32_v_f32m8(pI8 + 1, stride, vOut8i, vl);

               {
                  uint32_t elemInc = (uint32_t)(2U * n1 * (uint32_t)vl);
                  pI1 += elemInc;
                  pI2 += elemInc;
                  pI3 += elemInc;
                  pI4 += elemInc;
                  pI5 += elemInc;
                  pI6 += elemInc;
                  pI7 += elemInc;
                  pI8 += elemInc;
               }
               blkCnt -= (uint32_t)vl;
            }
         }
#else
         do
         {
            /*  index calculation for the input */
            i2 = i1 + n2;
            i3 = i2 + n2;
            i4 = i3 + n2;
            i5 = i4 + n2;
            i6 = i5 + n2;
            i7 = i6 + n2;
            i8 = i7 + n2;
            r1 = pSrc[2 * i1] + pSrc[2 * i5];
            r5 = pSrc[2 * i1] - pSrc[2 * i5];
            r2 = pSrc[2 * i2] + pSrc[2 * i6];
            r6 = pSrc[2 * i2] - pSrc[2 * i6];
            r3 = pSrc[2 * i3] + pSrc[2 * i7];
            r7 = pSrc[2 * i3] - pSrc[2 * i7];
            r4 = pSrc[2 * i4] + pSrc[2 * i8];
            r8 = pSrc[2 * i4] - pSrc[2 * i8];
            t1 = r1 - r3;
            r1 = r1 + r3;
            r3 = r2 - r4;
            r2 = r2 + r4;
            pSrc[2 * i1] = r1 + r2;
            r2 = r1 - r2;
            s1 = pSrc[2 * i1 + 1] + pSrc[2 * i5 + 1];
            s5 = pSrc[2 * i1 + 1] - pSrc[2 * i5 + 1];
            s2 = pSrc[2 * i2 + 1] + pSrc[2 * i6 + 1];
            s6 = pSrc[2 * i2 + 1] - pSrc[2 * i6 + 1];
            s3 = pSrc[2 * i3 + 1] + pSrc[2 * i7 + 1];
            s7 = pSrc[2 * i3 + 1] - pSrc[2 * i7 + 1];
            s4 = pSrc[2 * i4 + 1] + pSrc[2 * i8 + 1];
            s8 = pSrc[2 * i4 + 1] - pSrc[2 * i8 + 1];
            t2 = s1 - s3;
            s1 = s1 + s3;
            s3 = s2 - s4;
            s2 = s2 + s4;
            r1 = t1 + s3;
            t1 = t1 - s3;
            pSrc[2 * i1 + 1] = s1 + s2;
            s2 = s1 - s2;
            s1 = t2 - r3;
            t2 = t2 + r3;
            p1 = co5 * r2;
            p2 = si5 * s2;
            p3 = co5 * s2;
            p4 = si5 * r2;
            pSrc[2 * i5]     = p1 + p2;
            pSrc[2 * i5 + 1] = p3 - p4;
            p1 = co3 * r1;
            p2 = si3 * s1;
            p3 = co3 * s1;
            p4 = si3 * r1;
            pSrc[2 * i3]     = p1 + p2;
            pSrc[2 * i3 + 1] = p3 - p4;
            p1 = co7 * t1;
            p2 = si7 * t2;
            p3 = co7 * t2;
            p4 = si7 * t1;
            pSrc[2 * i7]     = p1 + p2;
            pSrc[2 * i7 + 1] = p3 - p4;
            r1 = (r6 - r8) * C81;
            r6 = (r6 + r8) * C81;
            s1 = (s6 - s8) * C81;
            s6 = (s6 + s8) * C81;
            t1 = r5 - r1;
            r5 = r5 + r1;
            r8 = r7 - r6;
            r7 = r7 + r6;
            t2 = s5 - s1;
            s5 = s5 + s1;
            s8 = s7 - s6;
            s7 = s7 + s6;
            r1 = r5 + s7;
            r5 = r5 - s7;
            r6 = t1 + s8;
            t1 = t1 - s8;
            s1 = s5 - r7;
            s5 = s5 + r7;
            s6 = t2 - r8;
            t2 = t2 + r8;
            p1 = co2 * r1;
            p2 = si2 * s1;
            p3 = co2 * s1;
            p4 = si2 * r1;
            pSrc[2 * i2]     = p1 + p2;
            pSrc[2 * i2 + 1] = p3 - p4;
            p1 = co8 * r5;
            p2 = si8 * s5;
            p3 = co8 * s5;
            p4 = si8 * r5;
            pSrc[2 * i8]     = p1 + p2;
            pSrc[2 * i8 + 1] = p3 - p4;
            p1 = co6 * r6;
            p2 = si6 * s6;
            p3 = co6 * s6;
            p4 = si6 * r6;
            pSrc[2 * i6]     = p1 + p2;
            pSrc[2 * i6 + 1] = p3 - p4;
            p1 = co4 * t1;
            p2 = si4 * t2;
            p3 = co4 * t2;
            p4 = si4 * t1;
            pSrc[2 * i4]     = p1 + p2;
            pSrc[2 * i4 + 1] = p3 - p4;

            i1 += n1;
         } while (i1 < fftLen);
#endif

         j++;
      } while (j < n2);

      twidCoefModifier <<= 3;
   } while (n2 > 7);
}
