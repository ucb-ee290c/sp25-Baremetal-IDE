# CMakeLists definitions for target `mfcclib`.

set(MFCC_LIB_DRIVER_SOURCES
  src/mfcc_driver.c
  src/mfcc_specialized_f32.c
  src/mfcc_specialized_f16.c
)

set(MFCC_LIB_NMSIS_BASIC_SOURCES
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

set(MFCC_LIB_NMSIS_STATISTICS_SOURCES
  src/nmsis/StatisticsFunctions/riscv_absmax_f32.c
  src/nmsis/StatisticsFunctions/riscv_absmax_f16.c
  src/nmsis/StatisticsFunctions/riscv_absmax_q15.c
  src/nmsis/StatisticsFunctions/riscv_absmax_q31.c
)

set(MFCC_LIB_NMSIS_FAST_MATH_SOURCES
  src/nmsis/FastMathFunctions/riscv_divide_q15.c
  src/nmsis/FastMathFunctions/riscv_divide_q31.c
  src/nmsis/FastMathFunctions/riscv_sqrt_q31.c
  src/nmsis/FastMathFunctions/riscv_vlog_f32.c
  src/nmsis/FastMathFunctions/riscv_vlog_f16.c
  src/nmsis/FastMathFunctions/riscv_vlog_q31.c
)

set(MFCC_LIB_NMSIS_COMPLEX_MATH_SOURCES
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_f32.c
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_f16.c
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_q15.c
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_q31.c
)

set(MFCC_LIB_NMSIS_MATRIX_SOURCES
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_f32.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_f16.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_q15.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_q31.c
)

set(MFCC_LIB_NMSIS_TRANSFORM_SOURCES
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

set(MFCC_LIB_NMSIS_COMMON_TABLES_SOURCES
  src/nmsis/CommonTables/riscv_common_tables.c
  src/nmsis/CommonTables/riscv_common_tables_f16.c
  src/nmsis/CommonTables/riscv_const_structs.c
  src/nmsis/CommonTables/riscv_const_structs_f16.c
)

add_library(mfcclib STATIC
  ${MFCC_LIB_DRIVER_SOURCES}
  ${MFCC_LIB_NMSIS_BASIC_SOURCES}
  ${MFCC_LIB_NMSIS_STATISTICS_SOURCES}
  ${MFCC_LIB_NMSIS_FAST_MATH_SOURCES}
  ${MFCC_LIB_NMSIS_COMPLEX_MATH_SOURCES}
  ${MFCC_LIB_NMSIS_MATRIX_SOURCES}
  ${MFCC_LIB_NMSIS_TRANSFORM_SOURCES}
  ${MFCC_LIB_NMSIS_COMMON_TABLES_SOURCES}
)

set(MFCC_LIB_F16_ASM_MODE 0 CACHE STRING
  "0=disable all f16 asm kernels, 1=enable all f16 asm kernels")
set_property(CACHE MFCC_LIB_F16_ASM_MODE PROPERTY STRINGS 0 1)

if(MFCC_LIB_F16_ASM_MODE EQUAL 1)
  target_sources(mfcclib PRIVATE src/asm/riscv_mult_f16_rvv.S)
  target_compile_definitions(mfcclib PRIVATE MFCC_F16_ASM_MULT=1)
  target_sources(mfcclib PRIVATE src/asm/riscv_rfft_stage_f16_rvv.S)
  target_compile_definitions(mfcclib PRIVATE MFCC_F16_ASM_RFFT_STAGE=1)
  target_sources(mfcclib PRIVATE src/asm/riscv_cfft_radix4by2_pre_f16_rvv.S)
  target_compile_definitions(mfcclib PRIVATE MFCC_F16_ASM_RADIX4BY2=1)
endif()

set(MFCC_LIB_F16_ACCUM_MODE 0 CACHE STRING
  "0=native f16 accumulation, 1=mixed f32 accumulation in MEL/DCT")
set_property(CACHE MFCC_LIB_F16_ACCUM_MODE PROPERTY STRINGS 0 1)
target_compile_definitions(mfcclib PRIVATE MFCC_F16_ACCUM_MODE=${MFCC_LIB_F16_ACCUM_MODE})

set(MFCC_LIB_VLOG_VEC_APPROX 1 CACHE STRING
  "0=scalar log kernels, 1=RVV polynomial vlog approximation")
set_property(CACHE MFCC_LIB_VLOG_VEC_APPROX PROPERTY STRINGS 0 1)
target_compile_definitions(mfcclib PRIVATE MFCC_VLOG_VEC_APPROX=${MFCC_LIB_VLOG_VEC_APPROX})

set(MFCC_LIB_SPECIALIZED_RUNTIME_CHECK 0 CACHE STRING
  "0=assume fixed specialized shape, 1=runtime shape check and fallback")
set_property(CACHE MFCC_LIB_SPECIALIZED_RUNTIME_CHECK PROPERTY STRINGS 0 1)
target_compile_definitions(mfcclib PRIVATE MFCC_SPECIALIZED_RUNTIME_CHECK=${MFCC_LIB_SPECIALIZED_RUNTIME_CHECK})

option(MFCC_LIB_F16_RADIX4_EXPERIMENTAL
  "Enable experimental RVV radix4 f16 butterfly path" OFF)
if(MFCC_LIB_F16_RADIX4_EXPERIMENTAL)
  target_compile_definitions(mfcclib PRIVATE RISCV_MFCC_F16_RADIX4_VEC_EXPERIMENTAL=1)
endif()

set(MFCC_LIB_HOTSPOT_SOURCES
  src/mfcc_specialized_f32.c
  src/nmsis/TransformFunctions/riscv_bitreversal2.c
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
set_source_files_properties(${MFCC_LIB_HOTSPOT_SOURCES} PROPERTIES COMPILE_OPTIONS "-O3")

set(MFCC_LIB_F16_HOTSPOT_SOURCES
  src/mfcc_specialized_f16.c
  src/nmsis/StatisticsFunctions/riscv_absmax_f16.c
  src/nmsis/BasicMathFunctions/riscv_mult_f16.c
  src/nmsis/BasicMathFunctions/riscv_offset_f16.c
  src/nmsis/BasicMathFunctions/riscv_scale_f16.c
  src/nmsis/TransformFunctions/riscv_mfcc_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_f16.c
  src/nmsis/TransformFunctions/riscv_cfft_radix4_f16.c
  src/nmsis/TransformFunctions/riscv_rfft_fast_f16.c
  src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_f16.c
  src/nmsis/MatrixFunctions/riscv_mat_vec_mult_f16.c
  src/nmsis/BasicMathFunctions/riscv_dot_prod_f16.c
  src/nmsis/FastMathFunctions/riscv_vlog_f16.c
)
set_source_files_properties(${MFCC_LIB_F16_HOTSPOT_SOURCES} PROPERTIES
  COMPILE_OPTIONS "-O3;-fno-strict-aliasing")

option(MFCC_LIB_F16_ULTRA_OPT "Enable aggressive f16 hotspot optimization" OFF)
if(MFCC_LIB_F16_ULTRA_OPT)
  target_compile_definitions(mfcclib PRIVATE MFCC_F16_ULTRA_OPT_ENABLED=1)
  set(MFCC_LIB_F16_ULTRA_SOURCES
    src/mfcc_specialized_f16.c
    src/nmsis/TransformFunctions/riscv_mfcc_f16.c
    src/nmsis/TransformFunctions/riscv_cfft_f16.c
    src/nmsis/TransformFunctions/riscv_cfft_radix4_f16.c
    src/nmsis/TransformFunctions/riscv_rfft_fast_f16.c
    src/nmsis/ComplexMathFunctions/riscv_cmplx_mag_f16.c
    src/nmsis/FastMathFunctions/riscv_vlog_f16.c
    src/nmsis/TransformFunctions/riscv_bitreversal2.c
  )
  set_source_files_properties(${MFCC_LIB_F16_ULTRA_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-O3;-fno-strict-aliasing;-fno-math-errno;-ffp-contract=fast")
endif()

target_include_directories(mfcclib PUBLIC include)
target_link_libraries(mfcclib PUBLIC m)
