#ifndef SVM_FUNCTIONS_F16_H_
#define SVM_FUNCTIONS_F16_H_

#include "riscv_math_types_f16.h"
#include "riscv_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"
#include "dsp/svm_defines.h"


#ifdef   __cplusplus
extern "C"
{
#endif

#if defined(RISCV_FLOAT16_SUPPORTED)

#define STEP(x) (x) <= 0 ? 0 : 1

/**
 * @defgroup groupSVM SVM Functions
 * This set of functions is implementing SVM classification on 2 classes.
 * The training must be done from scikit-learn. The parameters can be easily
 * generated from the scikit-learn object. Some examples are given in
 * DSP/Testing/PatternGeneration/SVM.py
 *
 * If more than 2 classes are needed, the functions in this folder 
 * will have to be used, as building blocks, to do multi-class classification.
 *
 * No multi-class classification is provided in this SVM folder.
 * 
 */



/**
 * @brief Instance structure for linear SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float16_t       intercept;              /**< Intercept */
  const float16_t *dualCoefficients;      /**< Dual coefficients */
  const float16_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
} riscv_svm_linear_instance_f16;


/**
 * @brief Instance structure for polynomial SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float16_t       intercept;              /**< Intercept */
  const float16_t *dualCoefficients;      /**< Dual coefficients */
  const float16_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
  int32_t         degree;                 /**< Polynomial degree */
  float16_t       coef0;                  /**< Polynomial constant */
  float16_t       gamma;                  /**< Gamma factor */
} riscv_svm_polynomial_instance_f16;


/**
 * @brief Instance structure for rbf SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float16_t       intercept;              /**< Intercept */
  const float16_t *dualCoefficients;      /**< Dual coefficients */
  const float16_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
  float16_t       gamma;                  /**< Gamma factor */
} riscv_svm_rbf_instance_f16;


/**
 * @brief Instance structure for sigmoid SVM prediction function.
 */
typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float16_t       intercept;              /**< Intercept */
  const float16_t *dualCoefficients;      /**< Dual coefficients */
  const float16_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
  float16_t       coef0;                  /**< Independent constant */
  float16_t       gamma;                  /**< Gamma factor */
} riscv_svm_sigmoid_instance_f16;


/**
 * @brief        SVM linear instance init function
 * @param[in]    S                      Parameters for SVM functions
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 */
void riscv_svm_linear_init_f16(riscv_svm_linear_instance_f16 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float16_t intercept,
  const float16_t *dualCoefficients,
  const float16_t *supportVectors,
  const int32_t  *classes);

/**
 * @brief SVM linear prediction
 * @param[in]    S          Pointer to an instance of the linear SVM structure.
 * @param[in]    in         Pointer to input vector
 * @param[out]   pResult    Decision value
 */
void riscv_svm_linear_predict_f16(const riscv_svm_linear_instance_f16 *S, 
   const float16_t * in, 
   int32_t * pResult);


/**
 * @brief        SVM polynomial instance init function
 * @param[in]    S                      points to an instance of the polynomial SVM structure.
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 * @param[in]    degree                 Polynomial degree
 * @param[in]    coef0                  coeff0 (scikit-learn terminology)
 * @param[in]    gamma                  gamma (scikit-learn terminology)
 */
void riscv_svm_polynomial_init_f16(riscv_svm_polynomial_instance_f16 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float16_t intercept,
  const float16_t *dualCoefficients,
  const float16_t *supportVectors,
  const int32_t   *classes,
  int32_t      degree,
  float16_t coef0,
  float16_t gamma
  );


/**
 * @brief SVM polynomial prediction
 * @param[in]    S          Pointer to an instance of the polynomial SVM structure.
 * @param[in]    in         Pointer to input vector
 * @param[out]   pResult    Decision value
 */
void riscv_svm_polynomial_predict_f16(const riscv_svm_polynomial_instance_f16 *S, 
   const float16_t * in, 
   int32_t * pResult);


/**
 * @brief        SVM radial basis function instance init function
 * @param[in]    S                      points to an instance of the polynomial SVM structure.
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 * @param[in]    gamma                  gamma (scikit-learn terminology)
 */
void riscv_svm_rbf_init_f16(riscv_svm_rbf_instance_f16 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float16_t intercept,
  const float16_t *dualCoefficients,
  const float16_t *supportVectors,
  const int32_t   *classes,
  float16_t gamma
  );


/**
 * @brief SVM rbf prediction
 * @param[in]    S         Pointer to an instance of the rbf SVM structure.
 * @param[in]    in        Pointer to input vector
 * @param[out]   pResult   decision value
 */
void riscv_svm_rbf_predict_f16(const riscv_svm_rbf_instance_f16 *S, 
   const float16_t * in, 
   int32_t * pResult);


/**
 * @brief        SVM sigmoid instance init function
 * @param[in]    S                      points to an instance of the rbf SVM structure.
 * @param[in]    nbOfSupportVectors     Number of support vectors
 * @param[in]    vectorDimension        Dimension of vector space
 * @param[in]    intercept              Intercept
 * @param[in]    dualCoefficients       Array of dual coefficients
 * @param[in]    supportVectors         Array of support vectors
 * @param[in]    classes                Array of 2 classes ID
 * @param[in]    coef0                  coeff0 (scikit-learn terminology)
 * @param[in]    gamma                  gamma (scikit-learn terminology)
 */
void riscv_svm_sigmoid_init_f16(riscv_svm_sigmoid_instance_f16 *S, 
  uint32_t nbOfSupportVectors,
  uint32_t vectorDimension,
  float16_t intercept,
  const float16_t *dualCoefficients,
  const float16_t *supportVectors,
  const int32_t   *classes,
  float16_t coef0,
  float16_t gamma
  );


/**
 * @brief SVM sigmoid prediction
 * @param[in]    S        Pointer to an instance of the rbf SVM structure.
 * @param[in]    in       Pointer to input vector
 * @param[out]   pResult  Decision value
 */
void riscv_svm_sigmoid_predict_f16(const riscv_svm_sigmoid_instance_f16 *S, 
   const float16_t * in, 
   int32_t * pResult);



#endif /*defined(RISCV_FLOAT16_SUPPORTED)*/
#ifdef   __cplusplus
}
#endif

#endif /* ifndef _SVM_FUNCTIONS_F16_H_ */
