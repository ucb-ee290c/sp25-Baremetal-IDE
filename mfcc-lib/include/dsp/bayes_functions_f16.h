#ifndef BAYES_FUNCTIONS_F16_H_
#define BAYES_FUNCTIONS_F16_H_

#include "riscv_math_types_f16.h"
#include "riscv_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"

#include "dsp/statistics_functions_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if defined(RISCV_FLOAT16_SUPPORTED)

/**
 * @brief Instance structure for Naive Gaussian Bayesian estimator.
 */
typedef struct
{
  uint32_t vectorDimension;  /**< Dimension of vector space */
  uint32_t numberOfClasses;  /**< Number of different classes  */
  const float16_t *theta;          /**< Mean values for the Gaussians */
  const float16_t *sigma;          /**< Variances for the Gaussians */
  const float16_t *classPriors;    /**< Class prior probabilities */
  float16_t epsilon;         /**< Additive value to variances */
} riscv_gaussian_naive_bayes_instance_f16;

/**
 * @brief Naive Gaussian Bayesian Estimator
 *
 * @param[in]  S                        points to a naive bayes instance structure
 * @param[in]  in                       points to the elements of the input vector.
 * @param[out] *pOutputProbabilities    points to a buffer of length numberOfClasses containing estimated probabilities
 * @param[out] *pBufferB                points to a temporary buffer of length numberOfClasses
 * @return The predicted class
 */
uint32_t riscv_gaussian_naive_bayes_predict_f16(const riscv_gaussian_naive_bayes_instance_f16 *S, 
   const float16_t * in, 
   float16_t *pOutputProbabilities,
   float16_t *pBufferB);

#endif /*defined(RISCV_FLOAT16_SUPPORTED)*/
#ifdef   __cplusplus
}
#endif

#endif /* ifndef _BAYES_FUNCTIONS_F16_H_ */
