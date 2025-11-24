# CMakeLists definitions for target `ope-bmarks`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(ope-bmarks
  src/main.c
  src/bench_cache.c
  src/bench_fill.c
  src/bench_impl.c
  src/bench_sizes.c
)

# Header Files
target_include_directories(ope-bmarks PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(ope-bmarks PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(ope-bmarks PRIVATE gcov)
endif()
