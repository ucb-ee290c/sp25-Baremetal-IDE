# CMakeLists definitions for target `can-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(can-test
  src/main.c
)

# Header Files
target_include_directories(can-test PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(can-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(can-test PRIVATE gcov)
endif()
