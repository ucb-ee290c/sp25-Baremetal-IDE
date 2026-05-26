# CMakeLists definitions for target `mobilenet`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mobilenet
  src/main.c
)

# Header Files
target_include_directories(mobilenet PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mobilenet PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(BUILD_NN_RVV)
  target_link_libraries(mobilenet PRIVATE nnrvv)
endif()

if (PROF_COV)
  target_link_libraries(mobilenet PRIVATE gcov)
endif()
