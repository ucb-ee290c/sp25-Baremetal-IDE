#ifndef SVM_DEFINES_H_
#define SVM_DEFINES_H_

/**
 * @brief Struct for specifying SVM Kernel
 */
typedef enum
{
    RISCV_ML_KERNEL_LINEAR = 0,
             /**< Linear kernel */
    RISCV_ML_KERNEL_POLYNOMIAL = 1,
             /**< Polynomial kernel */
    RISCV_ML_KERNEL_RBF = 2,
             /**< Radial Basis Function kernel */
    RISCV_ML_KERNEL_SIGMOID = 3
             /**< Sigmoid kernel */
} riscv_ml_kernel_type;

#endif
