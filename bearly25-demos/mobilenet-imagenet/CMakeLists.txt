# CMakeLists definitions for target `mobilenet-imagenet`.
# MobileNetV2 ImageNet Demo - Full float32 model generated from ONNX via onnx2c
#
# Note: The model.c file is very large (~130MB) due to embedded weights.
# This may require extended compile times and memory.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mobilenet-imagenet
  src/main.c
  src/model.c
)

# Header Files
target_include_directories(mobilenet-imagenet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optimization flags for the large model file
# -O3 and -ffast-math match the original Makefile settings
target_compile_options(mobilenet-imagenet PRIVATE
  -O3
  -ffast-math
  -Wno-unused-function
)

#################################
# Dependencies
#################################

target_link_libraries(mobilenet-imagenet PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

# Link math library (equivalent to -lm in Makefile)
target_link_libraries(mobilenet-imagenet PRIVATE m)

if(BUILD_VECNN)
  target_link_libraries(mobilenet-imagenet PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(mobilenet-imagenet PRIVATE gcov)
endif()
