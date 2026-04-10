if (NOT TARGET threadlib)
  message(FATAL_ERROR "bearly-kws requires THREAD_LIB=ON (threadlib target missing).")
endif()

add_executable(bearly-kws
  src/main.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/tinyspeech_model.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/tinyspeech_int8.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/tensor.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/misc.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/src/modules.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(bearly-kws PUBLIC include)
target_include_directories(bearly-kws PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc/include
  ${CMAKE_SOURCE_DIR}/thread-lib
)

target_compile_options(bearly-kws PRIVATE -O3)
target_compile_definitions(bearly-kws PRIVATE
  KWS_BEARLY_USE_THREADLIB=1
  TINYSPEECH_OPT_VARIANT_GEMM_RVV=1
  TINYSPEECH_CONV_FUSE_RELU=1
  TINYSPEECH_FUSE_POOL=1
  TINYSPEECH_OUTPUT_SOFTMAX=0
  TINYSPEECH_ENABLE_TRACE=0
  TINYSPEECH_REF_CHECK_STAGE_SUM=0
  TINYSPEECH_USE_TENSOR_ARENA=1
  TINYSPEECH_TENSOR_ARENA_SIZE=524288
  TINYSPEECH_INT8_PIPELINE=1
  TINYSPEECH_INT8_RVV_UKERNELS=1
  TINYSPEECH_INT8_MULTICORE=1
)

target_link_libraries(bearly-kws PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
  rocketcore
  threadlib
  m
)

if(PROF_COV)
  target_link_libraries(bearly-kws PRIVATE gcov)
endif()
