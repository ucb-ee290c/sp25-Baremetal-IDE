# CMakeLists definitions for target `convrev-demo`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(convrev-demo
  src/main.c
)

# Header Files
target_include_directories(convrev-demo PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(convrev-demo PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(convrev-demo PRIVATE gcov)
endif()
