# CMakeLists definitions for target `hello`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(hello
  src/main.c
)

# Header Files
target_include_directories(hello PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(hello PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(hello PRIVATE gcov)
endif()
