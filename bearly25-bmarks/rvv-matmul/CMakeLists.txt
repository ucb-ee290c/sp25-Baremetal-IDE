# CMakeLists definitions for target `ope-bmarks`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(rvv-matmul
  src/main.c
  src/f32_matmul_rvv.c
  src/i8_i16_matmul_rvv.c
  src/i8_i32_matmul_rvv.c
  src/bench_impl.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# Header Files
target_include_directories(rvv-matmul PUBLIC include)
target_include_directories(rvv-matmul PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

#################################
# Dependencies
#################################

target_link_libraries(rvv-matmul PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(rvv-matmul PRIVATE gcov)
endif()
