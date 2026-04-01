# CMakeLists definitions for target `tinyspeech-optimized4-test`.

include(CheckCSourceCompiles)
option(TINYSPEECH_OPT4_ENABLE_INT8 "Enable fixed-shape INT8 inference pipeline" ON)

set(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS}")
check_c_source_compiles("
  #include <stddef.h>
  #include <riscv_vector.h>
  int main(void) {
    size_t vl = __riscv_vsetvl_e16m8(8);
    vfloat16m8_t v = __riscv_vfmv_v_f_f16m8((_Float16)1.0f, vl);
    (void)v;
    return 0;
  }" TINYSPEECH_HAS_ZVFH_INTRINSICS)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

if (TINYSPEECH_HAS_ZVFH_INTRINSICS)
  set(TINYSPEECH_FP16_MIXED_VALUE 1)
  message(STATUS "tinyspeech-optimized4-test: fp16 RVV intrinsics available")
else()
  set(TINYSPEECH_FP16_MIXED_VALUE 0)
  message(STATUS "tinyspeech-optimized4-test: fp16 RVV intrinsics unavailable")
endif()

add_executable(tinyspeech-optimized4-test
  src/main.c
  src/tinyspeech_model.c
  src/tinyspeech_int8.c
  src/tensor.c
  src/misc.c
  src/modules.c
)

target_include_directories(tinyspeech-optimized4-test PUBLIC include)
target_include_directories(tinyspeech-optimized4-test PUBLIC ${CMAKE_SOURCE_DIR}/platform/dsp25/include)

target_compile_options(tinyspeech-optimized4-test PRIVATE -O3)
target_compile_definitions(tinyspeech-optimized4-test PRIVATE
  TINYSPEECH_OPT_VARIANT_GEMM_RVV=1
  TINYSPEECH_CONV_FUSE_RELU=1
  TINYSPEECH_FUSE_POOL=1
  TINYSPEECH_OUTPUT_SOFTMAX=0
  TINYSPEECH_ENABLE_TRACE=0
  TINYSPEECH_REF_CHECK_STAGE_SUM=0
  TINYSPEECH_USE_TENSOR_ARENA=1
  TINYSPEECH_TENSOR_ARENA_SIZE=524288
  TINYSPEECH_FP16_MIXED=${TINYSPEECH_FP16_MIXED_VALUE}
  TINYSPEECH_FP16_FC_ACCEL=0
  TINYSPEECH_INT8_PIPELINE=$<IF:$<BOOL:${TINYSPEECH_OPT4_ENABLE_INT8}>,1,0>
)

target_link_libraries(tinyspeech-optimized4-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if (PROF_COV)
  target_link_libraries(tinyspeech-optimized4-test PRIVATE gcov)
endif()
