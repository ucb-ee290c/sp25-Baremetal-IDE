# CMakeLists definitions for target `tinyspeech-optimized2-test`.

add_executable(tinyspeech-optimized2-test
  src/main.c
  src/tinyspeech_model.c
  src/tensor.c
  src/misc.c
  src/modules.c
)

target_include_directories(tinyspeech-optimized2-test PUBLIC include)
target_include_directories(tinyspeech-optimized2-test PUBLIC ${CMAKE_SOURCE_DIR}/platform/dsp25/include)

target_compile_options(tinyspeech-optimized2-test PRIVATE -O3)
target_compile_definitions(tinyspeech-optimized2-test PRIVATE
  TINYSPEECH_OPT_VARIANT_GEMM_RVV=1
  TINYSPEECH_CONV_FUSE_RELU=1
  TINYSPEECH_FUSE_POOL=1
  TINYSPEECH_OUTPUT_SOFTMAX=0
  TINYSPEECH_ENABLE_TRACE=0
  TINYSPEECH_REF_CHECK_STAGE_SUM=0
  TINYSPEECH_USE_TENSOR_ARENA=1
  TINYSPEECH_TENSOR_ARENA_SIZE=524288
)

target_link_libraries(tinyspeech-optimized2-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if (PROF_COV)
  target_link_libraries(tinyspeech-optimized2-test PRIVATE gcov)
endif()
