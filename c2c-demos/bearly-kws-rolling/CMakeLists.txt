set(KWS_BEARLY_ROLLING_USE_THREADLIB_VALUE 0)
if (TARGET threadlib)
  set(KWS_BEARLY_ROLLING_USE_THREADLIB_VALUE 1)
endif()

include(CheckCSourceCompiles)

set(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
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
  }" KWS_BEARLY_ROLLING_HAS_ZVFH_INTRINSICS)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

set(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
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
  }" KWS_BEARLY_ROLLING_HAS_RVV_I8_STRIDED_STORE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

set(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
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
  }" KWS_BEARLY_ROLLING_HAS_RVV_I8_CONTIG_PACK_STORE_API4)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT KWS_BEARLY_ROLLING_HAS_RVV_I8_CONTIG_PACK_STORE_API4)
  set(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
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
    }" KWS_BEARLY_ROLLING_HAS_RVV_I8_CONTIG_PACK_STORE_API3)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE}")
  unset(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE)
  unset(CMAKE_REQUIRED_FLAGS)
endif()

set(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
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
  }" KWS_BEARLY_ROLLING_HAS_RVV_VSMUL_API4)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE}")
unset(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT KWS_BEARLY_ROLLING_HAS_RVV_VSMUL_API4)
  set(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
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
    }" KWS_BEARLY_ROLLING_HAS_RVV_VSMUL_API3)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE}")
  unset(_KWS_BEARLY_ROLLING_OLD_TRY_COMPILE_TARGET_TYPE)
  unset(CMAKE_REQUIRED_FLAGS)
endif()

set(KWS_BEARLY_ROLLING_INT8_USE_VSSE8_STORE_VALUE 0)
if (KWS_BEARLY_ROLLING_HAS_RVV_I8_STRIDED_STORE)
  set(KWS_BEARLY_ROLLING_INT8_USE_VSSE8_STORE_VALUE 1)
endif()

set(KWS_BEARLY_ROLLING_INT8_USE_VSE8_PACK_STORE_VALUE 0)
set(KWS_BEARLY_ROLLING_RVV_NCLIP_API_VALUE 0)
if (KWS_BEARLY_ROLLING_HAS_RVV_I8_CONTIG_PACK_STORE_API4)
  set(KWS_BEARLY_ROLLING_INT8_USE_VSE8_PACK_STORE_VALUE 1)
  set(KWS_BEARLY_ROLLING_RVV_NCLIP_API_VALUE 4)
elseif (KWS_BEARLY_ROLLING_HAS_RVV_I8_CONTIG_PACK_STORE_API3)
  set(KWS_BEARLY_ROLLING_INT8_USE_VSE8_PACK_STORE_VALUE 1)
  set(KWS_BEARLY_ROLLING_RVV_NCLIP_API_VALUE 3)
endif()

set(KWS_BEARLY_ROLLING_INT8_USE_VSMUL_REQUANT_VALUE 0)
set(KWS_BEARLY_ROLLING_RVV_VSMUL_API_VALUE 0)
if (KWS_BEARLY_ROLLING_HAS_RVV_VSMUL_API4)
  set(KWS_BEARLY_ROLLING_INT8_USE_VSMUL_REQUANT_VALUE 1)
  set(KWS_BEARLY_ROLLING_RVV_VSMUL_API_VALUE 4)
elseif (KWS_BEARLY_ROLLING_HAS_RVV_VSMUL_API3)
  set(KWS_BEARLY_ROLLING_INT8_USE_VSMUL_REQUANT_VALUE 1)
  set(KWS_BEARLY_ROLLING_RVV_VSMUL_API_VALUE 3)
endif()

set(KWS_BEARLY_ROLLING_FP16_MIXED_VALUE 0)
if (KWS_BEARLY_ROLLING_HAS_ZVFH_INTRINSICS)
  set(KWS_BEARLY_ROLLING_FP16_MIXED_VALUE 1)
endif()

set(KWS_BEARLY_ROLLING_TINYSPEECH_INT8_MULTICORE_VALUE ${KWS_BEARLY_ROLLING_USE_THREADLIB_VALUE})

add_executable(bearly-kws-rolling
  src/main.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/tinyspeech_model.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/tinyspeech_int8.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/tensor.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/misc.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/modules.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(bearly-kws-rolling PUBLIC include)
target_include_directories(bearly-kws-rolling PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/include
  ${CMAKE_SOURCE_DIR}/thread-lib
)

target_compile_options(bearly-kws-rolling PRIVATE -O3)
target_compile_definitions(bearly-kws-rolling PRIVATE
  KWS_BEARLY_ROLLING_USE_THREADLIB=${KWS_BEARLY_ROLLING_USE_THREADLIB_VALUE}
  TINYSPEECH_OPT_VARIANT_GEMM_RVV=1
  TINYSPEECH_CONV_FUSE_RELU=1
  TINYSPEECH_FUSE_POOL=1
  TINYSPEECH_OUTPUT_SOFTMAX=0
  TINYSPEECH_ENABLE_TRACE=0
  TINYSPEECH_REF_CHECK_STAGE_SUM=0
  TINYSPEECH_USE_TENSOR_ARENA=1
  TINYSPEECH_TENSOR_ARENA_SIZE=524288
  TINYSPEECH_FP16_MIXED=${KWS_BEARLY_ROLLING_FP16_MIXED_VALUE}
  TINYSPEECH_FP16_FC_ACCEL=0
  TINYSPEECH_INT8_PIPELINE=1
  TINYSPEECH_INT8_USE_VSSE8_STORE=${KWS_BEARLY_ROLLING_INT8_USE_VSSE8_STORE_VALUE}
  TINYSPEECH_INT8_USE_VSE8_PACK_STORE=${KWS_BEARLY_ROLLING_INT8_USE_VSE8_PACK_STORE_VALUE}
  TINYSPEECH_RVV_NCLIP_API=${KWS_BEARLY_ROLLING_RVV_NCLIP_API_VALUE}
  TINYSPEECH_INT8_USE_VSMUL_REQUANT=${KWS_BEARLY_ROLLING_INT8_USE_VSMUL_REQUANT_VALUE}
  TINYSPEECH_RVV_VSMUL_API=${KWS_BEARLY_ROLLING_RVV_VSMUL_API_VALUE}
  TINYSPEECH_INT8_RVV_UKERNELS=1
  TINYSPEECH_INT8_MULTICORE=${KWS_BEARLY_ROLLING_TINYSPEECH_INT8_MULTICORE_VALUE}
)

target_link_libraries(bearly-kws-rolling PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
  rocketcore
  m
)
if (TARGET threadlib)
  target_link_libraries(bearly-kws-rolling PRIVATE threadlib)
endif()

if(PROF_COV)
  target_link_libraries(bearly-kws-rolling PRIVATE gcov)
endif()
