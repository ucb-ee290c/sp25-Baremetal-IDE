# CMakeLists for matmul-28ope-test.
# Tests gemm_i8_i32_28xm1(): 28 RVV rows + 8 OPE rows = 36 rows per call.

add_executable(matmul-28ope-test
  src/main.c
  # src/opervv.c  # disabled for OPE isolation test
)

target_include_directories(matmul-28ope-test PRIVATE
  ${CMAKE_SOURCE_DIR}/platform/bearly25/include
)

target_link_libraries(matmul-28ope-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if (PROF_COV)
  target_link_libraries(matmul-28ope-test PRIVATE gcov)
endif()
