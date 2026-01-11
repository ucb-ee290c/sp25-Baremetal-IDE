# CMakeLists definitions for target `benchmark_adjustable_128_2048`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(benchmark_adjustable_128_2048
  src/main.c
)

# Header Files
target_include_directories(benchmark_adjustable_128_2048 PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(benchmark_adjustable_128_2048 PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(benchmark_adjustable_128_2048 PRIVATE gcov)
endif()
