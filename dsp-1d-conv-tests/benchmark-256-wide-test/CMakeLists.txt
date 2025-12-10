# CMakeLists definitions for target `benchmark-256-wide-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(benchmark-256-wide-test
  src/main.c
)

# Header Files
target_include_directories(benchmark-256-wide-test PUBLIC include)

#################################
# Dependencies
#################################

# target_link_libraries(benchmark-256-wide-test PRIVATE 
#   -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
# )

target_link_libraries(benchmark-256-wide-test chip-config)


if (PROF_COV)
  target_link_libraries(benchmark-256-wide-test PRIVATE gcov)
endif()
