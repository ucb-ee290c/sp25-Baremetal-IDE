# CMakeLists definitions for target `mobilenet-imagenet-vector-quant`.
# MobileNetV2 ImageNet Demo - Quantized int8 model

#################################
# Build Configuration
#################################

# Source Files
add_executable(mobilenet-imagenet-vector-quant
  src/main_int8.c
  src/vec_bridge.c
  src/vec_extra_kernels.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# Header Files
target_include_directories(mobilenet-imagenet-vector-quant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(mobilenet-imagenet-vector-quant PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

# Optimization flags
target_compile_options(mobilenet-imagenet-vector-quant PRIVATE
  -O3
  -ffast-math
  -Wno-unused-function
)

#################################
# Dependencies
#################################

target_link_libraries(mobilenet-imagenet-vector-quant PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

# Link math library
target_link_libraries(mobilenet-imagenet-vector-quant PRIVATE m)

if(BUILD_VECNN)
  target_link_libraries(mobilenet-imagenet-vector-quant PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(mobilenet-imagenet-vector-quant PRIVATE gcov)
endif()
