# CMakeLists definitions for target `mnist_cnn`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mnist_cnn
  src/main.c
)

# Header Files
target_include_directories(mnist_cnn PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mnist_cnn PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
if(BUILD_NN_RVV)
  target_link_libraries(mnist_cnn PRIVATE nnrvv)
endif()

if (PROF_COV)
  target_link_libraries(mnist_cnn PRIVATE gcov)
endif()
