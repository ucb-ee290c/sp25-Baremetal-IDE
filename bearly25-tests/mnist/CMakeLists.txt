# CMakeLists definitions for target `mnist`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mnist
  src/main.c
)

# Header Files
target_include_directories(mnist PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(mnist PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
if(BUILD_VECNN)
  target_link_libraries(mnist PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(mnist PRIVATE gcov)
endif()
