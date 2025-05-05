# CMakeLists definitions for target `2d-conv-hello`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(2d-conv-hello
  src/main.c
)

# Header Files
target_include_directories(2d-conv-hello PUBLIC include)
target_include_directories(2d-conv-hello PUBLIC ${CMAKE_SOURCE_DIR}/platform/bearly25/include)

#################################
# Dependencies
#################################

target_link_libraries(2d-conv-hello PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if (PROF_COV)
  target_link_libraries(2d-conv-hello PRIVATE gcov)
endif()
