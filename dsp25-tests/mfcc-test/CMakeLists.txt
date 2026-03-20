# CMakeLists definitions for target `mfcc-test`.

#################################
# Build Configuration
#################################

set(MFCC_APP_SOURCES
  src/main.c
)

set(MFCC_NMSIS_BASIC_SOURCES
  src/nmsis/BasicMathFunctions/riscv_abs_q15.c
  src/nmsis/BasicMathFunctions/riscv_abs_q31.c
  src/nmsis/BasicMathFunctions/riscv_dot_prod_f32.c
  src/nmsis/BasicMathFunctions/riscv_dot_prod_f16.c
  src/nmsis/BasicMathFunctions/riscv_dot_prod_q15.c
  src/nmsis/BasicMathFunctions/riscv_dot_prod_q31.c
  src/nmsis/BasicMathFunctions/riscv_mult_f32.c
  src/nmsis/BasicMathFunctions/riscv_mult_f16.c
  src/nmsis/BasicMathFunctions/riscv_mult_q15.c
  src/nmsis/BasicMathFunctions/riscv_mult_q31.c
  src/nmsis/BasicMathFunctions/riscv_offset_f32.c
  src/nmsis/BasicMathFunctions/riscv_offset_f16.c
  src/nmsis/BasicMathFunctions/riscv_offset_q31.c
  src/nmsis/BasicMathFunctions/riscv_scale_f32.c
  src/nmsis/BasicMathFunctions/riscv_scale_f16.c
  src/nmsis/BasicMathFunctions/riscv_scale_q15.c
  src/nmsis/BasicMathFunctions/riscv_scale_q31.c
  src/nmsis/BasicMathFunctions/riscv_shift_q15.c
  src/nmsis/BasicMathFunctions/riscv_shift_q31.c
)

set(MFCC_NMSIS_STATISTICS_SOURCES
  src/nmsis/StatisticsFunctions/riscv_absmax_f32.c
  src/nmsis/StatisticsFunctions/riscv_absmax_f16.c
  src/nmsis/StatisticsFunctions/riscv_absmax_q15.c
  src/nmsis/StatisticsFunctions/riscv_absmax_q31.c
)

set(MFCC_NMSIS_FAST_MATH_SOURCES
  src/nmsis/FastMathFunctions/riscv_divide_q15.c
  src/nmsis/FastMathFunctions/riscv_divide_q31.c
  src/nmsis/FastMathFunctions/riscv_sqrt_q31.c
  src/nmsis/FastMathFunctions/riscv_vlog_f32.c
  src/nmsis/FastMathFunctions/riscv_vlog_f16.c
  src/nmsis/FastMathFunctions/riscv_vlog_q31.c
)

set(MFCC_NMSIS_COMPLEX_MATH_SOURCES
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_f32.c
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_f16.c
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_q15.c
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_q31.c
)

set(MFCC_NMSIS_MATRIX_SOURCES
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_f32.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_f16.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_q15.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_q31.c
)

set(MFCC_NMSIS_TRANSFORM_SOURCES
  src/nmsis/TransformFunctions/riscv_bitreversal.c
  src/nmsis/TransformFunctions/riscv_bitreversal2.c
  src/nmsis/TransformFunctions/riscv_bitreversal_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_f32.c
  src/nmsis/TransformFunctions/riscv_cfft_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_q15.c
  src/nmsis/TransformFunctions/riscv_cfft_q31.c
  src/nmsis/TransformFunctions/riscv_cfft_init_f32.c
  src/nmsis/TransformFunctions/riscv_cfft_init_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_init_q15.c
  src/nmsis/TransformFunctions/riscv_cfft_init_q31.c
  src/nmsis/TransformFunctions/riscv_cfft_radix8_f32.c
  src/nmsis/TransformFunctions/riscv_cfft_radix8_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_radix4_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_radix4_q15.c
  src/nmsis/TransformFunctions/riscv_cfft_radix4_q31.c
  src/nmsis/TransformFunctions/riscv_rfft_fast_f32.c
  src/nmsis/TransformFunctions/riscv_rfft_fast_f16.c
  src/nmsis/TransformFunctions/riscv_rfft_fast_init_f32.c
  src/nmsis/TransformFunctions/riscv_rfft_fast_init_f16.c
  src/nmsis/TransformFunctions/riscv_rfft_q15.c
  src/nmsis/TransformFunctions/riscv_rfft_q31.c
  src/nmsis/TransformFunctions/riscv_rfft_init_q15.c
  src/nmsis/TransformFunctions/riscv_rfft_init_q31.c
  src/nmsis/TransformFunctions/riscv_mfcc_f32.c
  src/nmsis/TransformFunctions/riscv_mfcc_f16.c
  src/nmsis/TransformFunctions/riscv_mfcc_q15.c
  src/nmsis/TransformFunctions/riscv_mfcc_q31.c
  src/nmsis/TransformFunctions/riscv_mfcc_init_f32.c
  src/nmsis/TransformFunctions/riscv_mfcc_init_f16.c
  src/nmsis/TransformFunctions/riscv_mfcc_init_q15.c
  src/nmsis/TransformFunctions/riscv_mfcc_init_q31.c
)

set(MFCC_NMSIS_COMMON_TABLES_SOURCES
  src/nmsis/CommonTables/riscv_common_tables.c
  src/nmsis/CommonTables/riscv_common_tables_f16.c
  src/nmsis/CommonTables/riscv_const_structs.c
  src/nmsis/CommonTables/riscv_const_structs_f16.c
)

add_executable(mfcc-test
  ${MFCC_APP_SOURCES}
  ${MFCC_NMSIS_BASIC_SOURCES}
  ${MFCC_NMSIS_STATISTICS_SOURCES}
  ${MFCC_NMSIS_FAST_MATH_SOURCES}
  ${MFCC_NMSIS_COMPLEX_MATH_SOURCES}
  ${MFCC_NMSIS_MATRIX_SOURCES}
  ${MFCC_NMSIS_TRANSFORM_SOURCES}
  ${MFCC_NMSIS_COMMON_TABLES_SOURCES}
)

set(MFCC_HOTSPOT_SOURCES
  src/nmsis/FastMathFunctions/riscv_vlog_q31.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_q31.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_q15.c
  src/nmsis/TransformFunctions/riscv_cfft_q31.c
  src/nmsis/TransformFunctions/riscv_cfft_q15.c
  src/nmsis/TransformFunctions/riscv_cfft_radix4_q31.c
  src/nmsis/TransformFunctions/riscv_cfft_radix4_q15.c
  src/nmsis/TransformFunctions/riscv_rfft_q31.c
  src/nmsis/TransformFunctions/riscv_rfft_q15.c
)
set_source_files_properties(${MFCC_HOTSPOT_SOURCES} PROPERTIES COMPILE_OPTIONS "-O3")

set(MFCC_F16_STABLE_SOURCES
  src/nmsis/TransformFunctions/riscv_cfft_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_radix4_f16.c
  src/nmsis/TransformFunctions/riscv_rfft_fast_f16.c
)
set_source_files_properties(${MFCC_F16_STABLE_SOURCES} PROPERTIES COMPILE_OPTIONS "-O1;-fno-strict-aliasing")

target_include_directories(mfcc-test PUBLIC include)
target_include_directories(mfcc-test PUBLIC ${CMAKE_SOURCE_DIR}/platform/dsp25/include)

target_link_libraries(mfcc-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if (PROF_COV)
  target_link_libraries(mfcc-test PRIVATE gcov)
endif()
