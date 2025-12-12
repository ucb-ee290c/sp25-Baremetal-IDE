# CMakeLists definitions for target `threadlib-tests`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(threadlib-tests
  src/main.c
  src/tests.c
  src/test_basic.c
  src/test_barrier.c
  src/test_sleep.c
  src/test_steal.c
)

# Header Files
target_include_directories(threadlib-tests PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(threadlib-tests PRIVATE 
  threadlib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(threadlib-tests PRIVATE gcov)
endif()
