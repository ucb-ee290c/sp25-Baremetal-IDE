# CMakeLists definitions for target `rvv-conv`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(rvv-conv
  src/main.c
  src/bench_impl.c
  src/bench_sizes.c
  src/vec-conv.S
  src/vec-conv-i8.S
  src/vec-conv5x5.S
  src/vec-conv5x5-i8.S
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# Header Files
target_include_directories(rvv-conv PUBLIC include)
target_include_directories(rvv-conv PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

#################################
# Dependencies
#################################

target_link_libraries(rvv-conv PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(rvv-conv PRIVATE gcov)
endif()
