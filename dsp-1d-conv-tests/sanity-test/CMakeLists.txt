# CMakeLists definitions for target `sanity-test`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(sanity-test
  src/main.c
)

# Header Files
target_include_directories(sanity-test PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(sanity-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(sanity-test PRIVATE gcov)
endif()
