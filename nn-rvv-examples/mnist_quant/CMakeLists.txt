# CMakeLists definitions for target `mnist_quant`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mnist_quant
  src/main.c
)

# Header Files
target_include_directories(mnist_quant PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mnist_quant PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
if(BUILD_NN_RVV)
  target_link_libraries(mnist_quant PRIVATE nnrvv)
endif()

if (PROF_COV)
  target_link_libraries(mnist_quant PRIVATE gcov)
endif()
