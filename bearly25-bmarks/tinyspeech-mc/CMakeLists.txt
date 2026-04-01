# CMakeLists definitions for target `tinyspeech-mc`.

if (NOT TARGET threadlib)
  message(FATAL_ERROR "tinyspeech-mc requires THREAD_LIB=ON (threadlib target missing).")
endif()

include(CheckCSourceCompiles)
option(TINYSPEECH_MC_ENABLE_INT8 "Enable fixed-shape INT8 inference pipeline" ON)
option(TINYSPEECH_MC_INT8_RVV_UKERNELS "Enable RVV int8 fixed-shape microkernels" ON)
option(TINYSPEECH_MC_ENABLE_MULTICORE "Enable 2-core split kernels for fixed INT8 path" ON)

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

set(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS}")
check_c_source_compiles("
  #include <stddef.h>
  #include <riscv_vector.h>
  int main(void) {
    size_t vl = __riscv_vsetvl_e32m4(8);
    vint32m4_t a = __riscv_vmv_v_x_i32m4(1, vl);
    vint16m2_t b = __riscv_vnclip_wx_i16m2(a, 0, vl);
    vint8m1_t c = __riscv_vnclip_wx_i8m1(b, 0, vl);
    int8_t out[16];
    __riscv_vsse8_v_i8m1(out, (ptrdiff_t)2, c, vl);
    return out[0];
  }" TINYSPEECH_HAS_RVV_I8_STRIDED_STORE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

set(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS}")
check_c_source_compiles("
  #include <stddef.h>
  #include <riscv_vector.h>
  int main(void) {
    size_t vl = __riscv_vsetvl_e32m4(8);
    vint32m4_t a = __riscv_vmv_v_x_i32m4(1, vl);
    vint16m2_t b = __riscv_vnclip_wx_i16m2(a, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t c = __riscv_vnclip_wx_i8m1(b, 0, __RISCV_VXRM_RNU, vl);
    int8_t out[16];
    __riscv_vse8_v_i8m1(out, c, vl);
    return out[0];
  }" TINYSPEECH_HAS_RVV_I8_CONTIG_PACK_STORE_API4)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT TINYSPEECH_HAS_RVV_I8_CONTIG_PACK_STORE_API4)
  set(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS}")
  check_c_source_compiles("
    #include <stddef.h>
    #include <riscv_vector.h>
    int main(void) {
      size_t vl = __riscv_vsetvl_e32m4(8);
      vint32m4_t a = __riscv_vmv_v_x_i32m4(1, vl);
      vint16m2_t b = __riscv_vnclip_wx_i16m2(a, 0, vl);
      vint8m1_t c = __riscv_vnclip_wx_i8m1(b, 0, vl);
      int8_t out[16];
      __riscv_vse8_v_i8m1(out, c, vl);
      return out[0];
    }" TINYSPEECH_HAS_RVV_I8_CONTIG_PACK_STORE_API3)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE}")
  unset(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE)
  unset(CMAKE_REQUIRED_FLAGS)
endif()

set(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS}")
check_c_source_compiles("
  #include <stddef.h>
  #include <riscv_vector.h>
  int main(void) {
    size_t vl = __riscv_vsetvl_e32m4(8);
    vint32m4_t a = __riscv_vmv_v_x_i32m4(123, vl);
    vint32m4_t q = __riscv_vsmul_vx_i32m4(a, 0x40000000, __RISCV_VXRM_RNU, vl);
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vget_v_i32m4_i32m1(q, 0));
  }" TINYSPEECH_HAS_RVV_VSMUL_API4)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT TINYSPEECH_HAS_RVV_VSMUL_API4)
  set(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS}")
  check_c_source_compiles("
    #include <stddef.h>
    #include <riscv_vector.h>
    int main(void) {
      size_t vl = __riscv_vsetvl_e32m4(8);
      vint32m4_t a = __riscv_vmv_v_x_i32m4(123, vl);
      vint32m4_t q = __riscv_vsmul_vx_i32m4(a, 0x40000000, vl);
      return __riscv_vmv_x_s_i32m1_i32(__riscv_vget_v_i32m4_i32m1(q, 0));
    }" TINYSPEECH_HAS_RVV_VSMUL_API3)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE}")
  unset(_TINYSPEECH_OLD_TRY_COMPILE_TARGET_TYPE)
  unset(CMAKE_REQUIRED_FLAGS)
endif()

if (TINYSPEECH_HAS_ZVFH_INTRINSICS)
  set(TINYSPEECH_FP16_MIXED_VALUE 1)
  message(STATUS "tinyspeech-mc: fp16 RVV intrinsics available")
else()
  set(TINYSPEECH_FP16_MIXED_VALUE 0)
  message(STATUS "tinyspeech-mc: fp16 RVV intrinsics unavailable")
endif()

if (TINYSPEECH_HAS_RVV_I8_STRIDED_STORE)
  set(TINYSPEECH_INT8_USE_VSSE8_STORE_VALUE 1)
  message(STATUS "tinyspeech-mc: RVV int8 strided-store intrinsics available")
else()
  set(TINYSPEECH_INT8_USE_VSSE8_STORE_VALUE 0)
  message(STATUS "tinyspeech-mc: RVV int8 strided-store intrinsics unavailable (scalar scatter fallback)")
endif()

set(TINYSPEECH_INT8_USE_VSE8_PACK_STORE_VALUE 0)
set(TINYSPEECH_RVV_NCLIP_API_VALUE 0)
if (TINYSPEECH_HAS_RVV_I8_CONTIG_PACK_STORE_API4)
  set(TINYSPEECH_INT8_USE_VSE8_PACK_STORE_VALUE 1)
  set(TINYSPEECH_RVV_NCLIP_API_VALUE 4)
  message(STATUS "tinyspeech-mc: RVV int8 contiguous pack/store intrinsics available (vnclip API=4)")
elseif (TINYSPEECH_HAS_RVV_I8_CONTIG_PACK_STORE_API3)
  set(TINYSPEECH_INT8_USE_VSE8_PACK_STORE_VALUE 1)
  set(TINYSPEECH_RVV_NCLIP_API_VALUE 3)
  message(STATUS "tinyspeech-mc: RVV int8 contiguous pack/store intrinsics available (vnclip API=3)")
else()
  message(STATUS "tinyspeech-mc: RVV int8 contiguous pack/store intrinsics unavailable")
endif()

set(TINYSPEECH_INT8_USE_VSMUL_REQUANT_VALUE 0)
set(TINYSPEECH_RVV_VSMUL_API_VALUE 0)
if (TINYSPEECH_HAS_RVV_VSMUL_API4)
  set(TINYSPEECH_INT8_USE_VSMUL_REQUANT_VALUE 1)
  set(TINYSPEECH_RVV_VSMUL_API_VALUE 4)
  message(STATUS "tinyspeech-mc: RVV i32 vsmul requant available (API=4)")
elseif (TINYSPEECH_HAS_RVV_VSMUL_API3)
  set(TINYSPEECH_INT8_USE_VSMUL_REQUANT_VALUE 1)
  set(TINYSPEECH_RVV_VSMUL_API_VALUE 3)
  message(STATUS "tinyspeech-mc: RVV i32 vsmul requant available (API=3)")
else()
  message(STATUS "tinyspeech-mc: RVV i32 vsmul requant unavailable (using widen-mul fallback)")
endif()

add_executable(tinyspeech-mc
  src/main.c
  src/tinyspeech_model.c
  src/tinyspeech_int8.c
  src/tensor.c
  src/misc.c
  src/modules.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(tinyspeech-mc PUBLIC include)
target_include_directories(tinyspeech-mc PRIVATE
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/thread-lib
)

target_compile_options(tinyspeech-mc PRIVATE -O3)
target_compile_definitions(tinyspeech-mc PRIVATE
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
  TINYSPEECH_INT8_PIPELINE=$<IF:$<BOOL:${TINYSPEECH_MC_ENABLE_INT8}>,1,0>
  TINYSPEECH_INT8_USE_VSSE8_STORE=${TINYSPEECH_INT8_USE_VSSE8_STORE_VALUE}
  TINYSPEECH_INT8_USE_VSE8_PACK_STORE=${TINYSPEECH_INT8_USE_VSE8_PACK_STORE_VALUE}
  TINYSPEECH_RVV_NCLIP_API=${TINYSPEECH_RVV_NCLIP_API_VALUE}
  TINYSPEECH_INT8_USE_VSMUL_REQUANT=${TINYSPEECH_INT8_USE_VSMUL_REQUANT_VALUE}
  TINYSPEECH_RVV_VSMUL_API=${TINYSPEECH_RVV_VSMUL_API_VALUE}
  TINYSPEECH_INT8_RVV_UKERNELS=$<IF:$<BOOL:${TINYSPEECH_MC_INT8_RVV_UKERNELS}>,1,0>
  TINYSPEECH_INT8_MULTICORE=$<IF:$<BOOL:${TINYSPEECH_MC_ENABLE_MULTICORE}>,1,0>
)

target_link_libraries(tinyspeech-mc PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
  rocketcore
  threadlib
  m
)

if (PROF_COV)
  target_link_libraries(tinyspeech-mc PRIVATE gcov)
endif()
