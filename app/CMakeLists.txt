# CMakeLists definitions for target `app`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(app
  src/main.c
)

# Header Files
target_include_directories(app PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(app PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(app PRIVATE gcov)
endif()
