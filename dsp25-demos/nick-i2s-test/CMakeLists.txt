# CMakeLists definitions for target `nick-i2s-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(nick-i2s-test
  src/main.c
)

# Header Files
target_include_directories(nick-i2s-test PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(nick-i2s-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(nick-i2s-test PRIVATE gcov)
endif()
