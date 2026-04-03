#define pack_weight_matrix pack_weight_matrix_f32
#define verify_matrix      verify_matrix_f32
#include "f32_matmul_rvv.c"
#undef pack_weight_matrix
#undef verify_matrix

#define pack_weight_matrix pack_weight_matrix_i8i16
#define verify_matrix      verify_matrix_i8i16
#include "i8_i16_matmul_rvv.c"
#undef pack_weight_matrix
#undef verify_matrix

#define pack_weight_matrix pack_weight_matrix_i8i32
#include "i8_i32_matmul_rvv.c"
#undef pack_weight_matrix

#define pack_weight_matrix pack_weight_matrix_i32
#include "i32_matmul_rvv.c"
#undef pack_weight_matrix

#define pack_weight_matrix pack_weight_matrix_i8i8
#include "i8_i8_matmul_rvv.c"
#undef pack_weight_matrix
