# CMakeLists definitions for target `test-multiple-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(test-multiple-test
  src/main.c
)

# Header Files
target_include_directories(test-multiple-test PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(test-multiple-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(test-multiple-test PRIVATE gcov)
endif()
