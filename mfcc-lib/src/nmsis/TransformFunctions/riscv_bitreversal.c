#include "dsp/transform_functions.h"
#include "riscv_common_tables.h"

void riscv_bitreversal_f32(
        float32_t * pSrc,
        uint16_t fftSize,
        uint16_t bitRevFactor,
  const uint16_t * pBitRevTab);

/**
  @brief         In-place floating-point bit reversal function.
  @param[in,out] pSrc         points to in-place floating-point data buffer
  @param[in]     fftSize      length of FFT
  @param[in]     bitRevFactor bit reversal modifier that supports different size FFTs with the same bit reversal table
  @param[in]     pBitRevTab   points to bit reversal table
 */

RISCV_DSP_ATTRIBUTE void riscv_bitreversal_f32(
        float32_t * pSrc,
        uint16_t fftSize,
        uint16_t bitRevFactor,
  const uint16_t * pBitRevTab)
{
   uint16_t fftLenBy2, fftLenBy2p1;
   uint16_t i, j;
   float32_t in;

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

void riscv_bitreversal_q31(
        q31_t * pSrc,
        uint32_t fftLen,
        uint16_t bitRevFactor,
  const uint16_t * pBitRevTab);

/**
  @brief         In-place Q31 bit reversal function.
  @param[in,out] pSrc         points to in-place Q31 data buffer.
  @param[in]     fftLen       length of FFT.
  @param[in]     bitRevFactor bit reversal modifier that supports different size FFTs with the same bit reversal table
  @param[in]     pBitRevTab   points to bit reversal table
*/

RISCV_DSP_ATTRIBUTE void riscv_bitreversal_q31(
        q31_t * pSrc,
        uint32_t fftLen,
        uint16_t bitRevFactor,
  const uint16_t * pBitRevTab)
{
   uint32_t fftLenBy2, fftLenBy2p1, i, j;
   q31_t in;

   /*  Initializations      */
   j = 0U;
   fftLenBy2 = fftLen / 2U;
   fftLenBy2p1 = (fftLen / 2U) + 1U;

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

      /*  Updating the bit reversal index depending on the fft length */
      pBitRevTab += bitRevFactor;
   }
}

void riscv_bitreversal_q15(
        q15_t * pSrc16,
        uint32_t fftLen,
        uint16_t bitRevFactor,
  const uint16_t * pBitRevTab);


/**
  @brief         In-place Q15 bit reversal function.
  @param[in,out] pSrc16       points to in-place Q15 data buffer
  @param[in]     fftLen       length of FFT
  @param[in]     bitRevFactor bit reversal modifier that supports different size FFTs with the same bit reversal table
  @param[in]     pBitRevTab   points to bit reversal table
*/

RISCV_DSP_ATTRIBUTE void riscv_bitreversal_q15(
        q15_t * pSrc16,
        uint32_t fftLen,
        uint16_t bitRevFactor,
  const uint16_t * pBitRevTab)
{
   q31_t *pSrc = (q31_t *) pSrc16;
   q31_t in;
   uint32_t fftLenBy2, fftLenBy2p1;
   uint32_t i, j;

   /*  Initializations */
   j = 0U;
   fftLenBy2 = fftLen / 2U;
   fftLenBy2p1 = (fftLen / 2U) + 1U;

   /* Bit Reversal Implementation */
   for (i = 0U; i <= (fftLenBy2 - 2U); i += 2U)
   {
      if (i < j)
      {
         /*  pSrc[i] <-> pSrc[j]; */
         /*  pSrc[i+1U] <-> pSrc[j+1U] */
         in = pSrc[i];
         pSrc[i] = pSrc[j];
         pSrc[j] = in;

         /*  pSrc[i + fftLenBy2p1] <-> pSrc[j + fftLenBy2p1];  */
         /*  pSrc[i + fftLenBy2p1+1U] <-> pSrc[j + fftLenBy2p1+1U] */
         in = pSrc[i + fftLenBy2p1];
         pSrc[i + fftLenBy2p1] = pSrc[j + fftLenBy2p1];
         pSrc[j + fftLenBy2p1] = in;
      }

      /*  pSrc[i+1U] <-> pSrc[j+fftLenBy2];         */
      /*  pSrc[i+2] <-> pSrc[j+fftLenBy2+1U]  */
      in = pSrc[i + 1U];
      pSrc[i + 1U] = pSrc[j + fftLenBy2];
      pSrc[j + fftLenBy2] = in;

      /*  Reading the index for the bit reversal */
      j = *pBitRevTab;

      /*  Updating the bit reversal index depending on the fft length  */
      pBitRevTab += bitRevFactor;
   }
}
