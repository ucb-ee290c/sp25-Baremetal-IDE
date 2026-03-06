# CMakeLists definitions for target `matmul-rvv-ope`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(matmul-rvv-ope
  src/main.c
  src/bench_cache.c
  src/bench_run.c
  src/i8_i32_matmul.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# Header Files
target_include_directories(matmul-rvv-ope PUBLIC include)
target_include_directories(matmul-rvv-ope PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

#################################
# Dependencies
#################################

target_link_libraries(matmul-rvv-ope PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(matmul-rvv-ope PRIVATE gcov)
endif()
