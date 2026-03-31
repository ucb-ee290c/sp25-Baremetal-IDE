# CMakeLists for matmul-64ope-test.
# Tests 64×64 int8 matmul: 8 OPE rows + 56 RVV rows, interleaved.

add_executable(matmul-64ope-test
  src/main.c
  src/opervv.c
  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/rvv-matmul/src/i8_i32_matmul_rvv.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(matmul-64ope-test PRIVATE
  ${CMAKE_SOURCE_DIR}/platform/bearly25/include
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_link_libraries(matmul-64ope-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if (PROF_COV)
  target_link_libraries(matmul-64ope-test PRIVATE gcov)
endif()
