# CMakeLists for matmul-rvv-test.
# Tests i8_i32_matmul() (the RVV A^T*B kernel) at 64x64.

add_executable(matmul-rvv-test
  src/main.c
)

target_include_directories(matmul-rvv-test PRIVATE
  ${CMAKE_SOURCE_DIR}/platform/bearly25/include
)

target_link_libraries(matmul-rvv-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if (PROF_COV)
  target_link_libraries(matmul-rvv-test PRIVATE gcov)
endif()
