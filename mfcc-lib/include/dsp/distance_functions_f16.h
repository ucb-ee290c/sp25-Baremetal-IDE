#ifndef DISTANCE_FUNCTIONS_F16_H_
#define DISTANCE_FUNCTIONS_F16_H_

#include "riscv_math_types_f16.h"
#include "riscv_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"

/* 6.14 bug */

#include "dsp/statistics_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"

#include "dsp/fast_math_functions_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if defined(RISCV_FLOAT16_SUPPORTED)

/**
 * @brief        Euclidean distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_euclidean_distance_f16(const float16_t *pA,const float16_t *pB, uint32_t blockSize);


/**
 * @brief        Bray-Curtis distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_braycurtis_distance_f16(const float16_t *pA,const float16_t *pB, uint32_t blockSize);

/**
 * @brief        Canberra distance between two vectors
 *
 * This function may divide by zero when samples pA[i] and pB[i] are both zero.
 * The result of the computation will be correct. So the division per zero may be
 * ignored.
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_canberra_distance_f16(const float16_t *pA,const float16_t *pB, uint32_t blockSize);


/**
 * @brief        Chebyshev distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_chebyshev_distance_f16(const float16_t *pA,const float16_t *pB, uint32_t blockSize);


/**
 * @brief        Cityblock (Manhattan) distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_cityblock_distance_f16(const float16_t *pA,const float16_t *pB, uint32_t blockSize);


/**
 * @brief        Correlation distance between two vectors
 *
 * The input vectors are modified in place !
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_correlation_distance_f16(float16_t *pA,float16_t *pB, uint32_t blockSize);


/**
 * @brief        Cosine distance between two vectors
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_cosine_distance_f16(const float16_t *pA,const float16_t *pB, uint32_t blockSize);


/**
 * @brief        Jensen-Shannon distance between two vectors
 *
 * This function is assuming that elements of second vector are > 0
 * and 0 only when the corresponding element of first vector is 0.
 * Otherwise the result of the computation does not make sense
 * and for speed reasons, the cases returning NaN or Infinity are not
 * managed.
 *
 * When the function is computing x log (x / y) with x 0 and y 0,
 * it will compute the right value (0) but a division per zero will occur
 * and shoudl be ignored in client code.
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_jensenshannon_distance_f16(const float16_t *pA,const float16_t *pB,uint32_t blockSize);


/**
 * @brief        Minkowski distance between two vectors
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    n          Norm order (>= 2)
 * @param[in]    blockSize  vector length
 * @return distance
 */
float16_t riscv_minkowski_distance_f16(const float16_t *pA,const float16_t *pB, int32_t order, uint32_t blockSize);


#endif /*defined(RISCV_FLOAT16_SUPPORTED)*/
#ifdef   __cplusplus
}
#endif

#endif /* ifndef _DISTANCE_FUNCTIONS_F16_H_ */
