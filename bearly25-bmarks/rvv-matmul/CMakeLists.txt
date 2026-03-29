# CMakeLists definitions for target `rvv-matmul`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(rvv-matmul
  src/main.c
  src/bench_cache.c
  src/bench_impl.c
  src/bench_run.c
  src/bench_sizes.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/hthread.c
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

target_link_libraries(rvv-matmul PRIVATE chip-config)
target_link_libraries(rvv-matmul PRIVATE rocketcore)

if (PROF_COV)
  target_link_libraries(rvv-matmul PRIVATE gcov)
endif()
