# CMakeLists definitions for target `mobilenet-imagenet-quant`.
# MobileNetV2 ImageNet Demo - Quantized int8 model

#################################
# Build Configuration
#################################

# Source Files
add_executable(mobilenet-imagenet-quant
  src/main_int8.c
)

# Header Files
target_include_directories(mobilenet-imagenet-quant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optimization flags
target_compile_options(mobilenet-imagenet-quant PRIVATE
  -O3
  -ffast-math
  -Wno-unused-function
)

#################################
# Dependencies
#################################

target_link_libraries(mobilenet-imagenet-quant PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

# Link math library
target_link_libraries(mobilenet-imagenet-quant PRIVATE m)

if(BUILD_VECNN)
  target_link_libraries(mobilenet-imagenet-quant PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(mobilenet-imagenet-quant PRIVATE gcov)
endif()
