# CMakeLists definitions for target `queue-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(queue-test
  src/main.c
)

# Header Files
target_include_directories(queue-test PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(queue-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(queue-test PRIVATE gcov)
endif()
