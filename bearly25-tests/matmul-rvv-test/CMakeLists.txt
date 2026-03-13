# CMakeLists for matmul-rvv-test.
# Tests i8_i32_matmul() (the RVV A^T*B kernel).
#
# Build options:
#   -DMATMUL_DUAL_CORE=ON   enable two-hart 128x128 split (default: OFF = 64x64 single-core)

option(MATMUL_DUAL_CORE "Build dual-core 128x128 matmul test" OFF)

add_executable(matmul-rvv-test
  src/main.c
)

target_include_directories(matmul-rvv-test PRIVATE
  ${CMAKE_SOURCE_DIR}/platform/bearly25/include
)

if (MATMUL_DUAL_CORE)
  target_compile_definitions(matmul-rvv-test PRIVATE DUAL_CORE)
endif()

target_link_libraries(matmul-rvv-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if (PROF_COV)
  target_link_libraries(matmul-rvv-test PRIVATE gcov)
endif()
