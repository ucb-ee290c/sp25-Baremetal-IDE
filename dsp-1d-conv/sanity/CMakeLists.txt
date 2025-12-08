# CMakeLists definitions for target `sanity`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(sanity
  src/main.c
)

# Header Files
target_include_directories(sanity PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(sanity PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(sanity PRIVATE gcov)
endif()
