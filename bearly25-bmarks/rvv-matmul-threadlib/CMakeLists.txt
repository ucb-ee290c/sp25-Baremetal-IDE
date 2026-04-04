# CMakeLists definitions for target `rvv-matmul-threadlib`.

if (NOT TARGET threadlib)
  message(FATAL_ERROR "rvv-matmul-threadlib requires THREAD_LIB=ON (threadlib target missing).")
endif()

#################################
# Build Configuration
#################################

# Source Files
add_executable(rvv-matmul-threadlib
  src/main.c
  src/bench_cache.c
  src/bench_impl.c
  src/bench_run.c
  src/bench_sizes.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# Header Files
target_include_directories(rvv-matmul-threadlib PUBLIC include)
target_include_directories(rvv-matmul-threadlib PRIVATE
  ${CMAKE_SOURCE_DIR}/thread-lib
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

#################################
# Dependencies
#################################

target_link_libraries(rvv-matmul-threadlib PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

target_link_libraries(rvv-matmul-threadlib PRIVATE chip-config)
target_link_libraries(rvv-matmul-threadlib PRIVATE rocketcore)
target_link_libraries(rvv-matmul-threadlib PRIVATE threadlib)

if (PROF_COV)
  target_link_libraries(rvv-matmul-threadlib PRIVATE gcov)
endif()
