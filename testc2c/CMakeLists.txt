# CMakeLists definitions for target `testc2c`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(testc2c
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# Header Files
target_include_directories(testc2c PUBLIC include)
target_include_directories(testc2c PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

#################################
# Dependencies
#################################

target_link_libraries(testc2c PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(testc2c PRIVATE gcov)
endif()
