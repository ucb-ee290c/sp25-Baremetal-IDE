# CMakeLists definitions for target `mnist_cnn_quant`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mnist_cnn_quant
  src/main.c
)

# Header Files
target_include_directories(mnist_cnn_quant PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mnist_cnn_quant PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
if(BUILD_VECNN)
  target_link_libraries(mnist_cnn_quant PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(mnist_cnn_quant PRIVATE gcov)
endif()
