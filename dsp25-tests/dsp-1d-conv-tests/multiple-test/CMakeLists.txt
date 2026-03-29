# CMakeLists definitions for target `multiple-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(multiple-test
  src/main.c
)

# Header Files
target_include_directories(multiple-test PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(multiple-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(multiple-test PRIVATE gcov)
endif()
