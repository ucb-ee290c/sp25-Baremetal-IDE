# CMakeLists definitions for target `mnist_quant2`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mnist_quant2
  src/main.c
)

# Header Files
target_include_directories(mnist_quant2 PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mnist_quant2 PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
if(BUILD_NN_RVV)
  target_link_libraries(mnist_quant2 PRIVATE nnrvv)
endif()

if (PROF_COV)
  target_link_libraries(mnist_quant2 PRIVATE gcov)
endif()
