# CMakeLists definitions for target `ope-tests`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(ope-tests
  src/main.c
)

# Header Files
target_include_directories(ope-tests PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(ope-tests PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(ope-tests PRIVATE gcov)
endif()
