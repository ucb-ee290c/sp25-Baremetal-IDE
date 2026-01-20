# CMakeLists definitions for target `convrev`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(convrev
  src/main.c
)

# Header Files
target_include_directories(convrev PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(convrev PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(convrev PRIVATE gcov)
endif()
