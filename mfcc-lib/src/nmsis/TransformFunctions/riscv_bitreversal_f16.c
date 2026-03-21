#include "dsp/transform_functions_f16.h"

/*
* @brief  In-place bit reversal function.
* @param[in, out] *pSrc        points to the in-place buffer of floating-point data type.
* @param[in]      fftSize      length of the FFT.
* @param[in]      bitRevFactor bit reversal modifier that supports different size FFTs with the same bit reversal table.
* @param[in]      *pBitRevTab  points to the bit reversal table.
*/

#if defined(RISCV_FLOAT16_SUPPORTED)

RISCV_DSP_ATTRIBUTE void riscv_bitreversal_f16(
float16_t * pSrc,
uint16_t fftSize,
uint16_t bitRevFactor,
const uint16_t * pBitRevTab)
{
   uint16_t fftLenBy2, fftLenBy2p1;
   uint16_t i, j;
   float16_t in;

   /*  Initializations */
   j = 0U;
   fftLenBy2 = fftSize >> 1U;
   fftLenBy2p1 = (fftSize >> 1U) + 1U;

   /* Bit Reversal Implementation */
   for (i = 0U; i <= (fftLenBy2 - 2U); i += 2U)
   {
      if (i < j)
      {
         /*  pSrc[i] <-> pSrc[j]; */
         in = pSrc[2U * i];
         pSrc[2U * i] = pSrc[2U * j];
         pSrc[2U * j] = in;

         /*  pSrc[i+1U] <-> pSrc[j+1U] */
         in = pSrc[(2U * i) + 1U];
         pSrc[(2U * i) + 1U] = pSrc[(2U * j) + 1U];
         pSrc[(2U * j) + 1U] = in;

         /*  pSrc[i+fftLenBy2p1] <-> pSrc[j+fftLenBy2p1] */
         in = pSrc[2U * (i + fftLenBy2p1)];
         pSrc[2U * (i + fftLenBy2p1)] = pSrc[2U * (j + fftLenBy2p1)];
         pSrc[2U * (j + fftLenBy2p1)] = in;

         /*  pSrc[i+fftLenBy2p1+1U] <-> pSrc[j+fftLenBy2p1+1U] */
         in = pSrc[(2U * (i + fftLenBy2p1)) + 1U];
         pSrc[(2U * (i + fftLenBy2p1)) + 1U] =
         pSrc[(2U * (j + fftLenBy2p1)) + 1U];
         pSrc[(2U * (j + fftLenBy2p1)) + 1U] = in;

      }

      /*  pSrc[i+1U] <-> pSrc[j+1U] */
      in = pSrc[2U * (i + 1U)];
      pSrc[2U * (i + 1U)] = pSrc[2U * (j + fftLenBy2)];
      pSrc[2U * (j + fftLenBy2)] = in;

      /*  pSrc[i+2U] <-> pSrc[j+2U] */
      in = pSrc[(2U * (i + 1U)) + 1U];
      pSrc[(2U * (i + 1U)) + 1U] = pSrc[(2U * (j + fftLenBy2)) + 1U];
      pSrc[(2U * (j + fftLenBy2)) + 1U] = in;

      /*  Reading the index for the bit reversal */
      j = *pBitRevTab;

      /*  Updating the bit reversal index depending on the fft length  */
      pBitRevTab += bitRevFactor;
   }
}
#endif /* #if defined(RISCV_FLOAT16_SUPPORTED) */
