# CMakeLists definitions for target `simple-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(simple-test
  src/main.c
)

# Header Files
target_include_directories(simple-test PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(simple-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(simple-test PRIVATE gcov)
endif()
