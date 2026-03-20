#ifndef BAYES_FUNCTIONS_H_
#define BAYES_FUNCTIONS_H_

#include "riscv_math_types.h"
#include "riscv_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"

#include "dsp/statistics_functions.h"

/**
 * @defgroup groupBayes Bayesian estimators
 *
 * Implement the naive gaussian Bayes estimator.
 * The training must be done from scikit-learn.
 *
 * The parameters can be easily
 * generated from the scikit-learn object. Some examples are given in
 * DSP/Testing/PatternGeneration/Bayes.py
 */

#ifdef   __cplusplus
extern "C"
{
#endif

/**
 * @brief Instance structure for Naive Gaussian Bayesian estimator.
 */
typedef struct
{
  uint32_t vectorDimension;  /**< Dimension of vector space */
  uint32_t numberOfClasses;  /**< Number of different classes  */
  const float32_t *theta;          /**< Mean values for the Gaussians */
  const float32_t *sigma;          /**< Variances for the Gaussians */
  const float32_t *classPriors;    /**< Class prior probabilities */
  float32_t epsilon;         /**< Additive value to variances */
} riscv_gaussian_naive_bayes_instance_f32;

/**
 * @brief Naive Gaussian Bayesian Estimator
 *
 * @param[in]  S                        points to a naive bayes instance structure
 * @param[in]  in                       points to the elements of the input vector.
 * @param[out] *pOutputProbabilities    points to a buffer of length numberOfClasses containing estimated probabilities
 * @param[out] *pBufferB                points to a temporary buffer of length numberOfClasses
 * @return The predicted class
 */
uint32_t riscv_gaussian_naive_bayes_predict_f32(const riscv_gaussian_naive_bayes_instance_f32 *S, 
   const float32_t * in, 
   float32_t *pOutputProbabilities,
   float32_t *pBufferB);


#ifdef   __cplusplus
}
#endif

#endif /* ifndef _BAYES_FUNCTIONS_H_ */
