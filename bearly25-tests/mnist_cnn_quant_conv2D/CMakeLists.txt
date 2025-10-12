# CMakeLists definitions for target `mnist_cnn_quant_conv2D`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mnist_cnn_quant_conv2D
  src/main.c
)

# Header Files
target_include_directories(mnist_cnn_quant_conv2D PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mnist_cnn_quant_conv2D PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
if(BUILD_VECNN)
  target_link_libraries(mnist_cnn_quant_conv2D PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(mnist_cnn_quant_conv2D PRIVATE gcov)
endif()
