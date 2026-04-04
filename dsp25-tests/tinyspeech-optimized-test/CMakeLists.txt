# CMakeLists definitions for target `tinyspeech-optimized-test`.

add_executable(tinyspeech-optimized-test
  src/main.c
  src/tinyspeech_model.c
  src/tensor.c
  src/misc.c
  src/modules.c
)

target_include_directories(tinyspeech-optimized-test PUBLIC include)
target_include_directories(tinyspeech-optimized-test PUBLIC ${CMAKE_SOURCE_DIR}/platform/dsp25/include)

target_compile_options(tinyspeech-optimized-test PRIVATE -O3)
target_compile_definitions(tinyspeech-optimized-test PRIVATE TINYSPEECH_OPT_VARIANT_DIRECT_RVV=1)

target_link_libraries(tinyspeech-optimized-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if (PROF_COV)
  target_link_libraries(tinyspeech-optimized-test PRIVATE gcov)
endif()
