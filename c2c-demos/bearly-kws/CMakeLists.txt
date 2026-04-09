add_executable(bearly-kws
  src/main.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-sc/src/tinyspeech_model.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-sc/src/tinyspeech_int8.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-sc/src/tensor.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-sc/src/misc.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-sc/src/modules.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(bearly-kws PUBLIC include)
target_include_directories(bearly-kws PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-sc/include
)

target_compile_options(bearly-kws PRIVATE -O3)
target_compile_definitions(bearly-kws PRIVATE
  TINYSPEECH_CONV_FUSE_RELU=1
  TINYSPEECH_FUSE_POOL=1
  TINYSPEECH_OUTPUT_SOFTMAX=0
  TINYSPEECH_ENABLE_TRACE=0
  TINYSPEECH_REF_CHECK_STAGE_SUM=0
  TINYSPEECH_USE_TENSOR_ARENA=1
  TINYSPEECH_TENSOR_ARENA_SIZE=524288
  TINYSPEECH_INT8_PIPELINE=1
  TINYSPEECH_INT8_RVV_UKERNELS=1
)

target_link_libraries(bearly-kws PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(bearly-kws PRIVATE gcov)
endif()
