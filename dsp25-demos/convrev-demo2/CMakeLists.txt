# CMakeLists definitions for target `convrev-demo2`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(convrev-demo2
  src/main.c
)

# Header Files
target_include_directories(convrev-demo2 PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(convrev-demo2 PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(convrev-demo2 PRIVATE gcov)
endif()
