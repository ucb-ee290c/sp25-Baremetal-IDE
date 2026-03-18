# CMakeLists definitions for target `mobilenet_tests`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mobilenet_tests
  src/main.c
)

# Header Files
target_include_directories(mobilenet_tests PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mobilenet_tests PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
if(BUILD_VECNN)
  target_link_libraries(mobilenet_tests PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(mobilenet_tests PRIVATE gcov)
endif()
