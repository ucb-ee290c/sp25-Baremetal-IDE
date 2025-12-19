# CMakeLists definitions for target `threadlib_tests`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(threadlib_tests
  src/main.c
  src/tests.c
  src/test_basic.c
  src/test_barrier.c
  src/test_sleep.c
  src/test_steal.c
)

# Header Files
target_include_directories(threadlib_tests PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(threadlib_tests PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  threadlib
)

if (PROF_COV)
  target_link_libraries(threadlib_tests PRIVATE gcov)
endif()
