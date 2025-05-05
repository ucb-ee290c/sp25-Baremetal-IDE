# CMakeLists definitions for target `mm_basic`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(mm_basic
  src/main.c
)

# Header Files
target_include_directories(mm_basic PUBLIC include)
target_include_directories(mm_basic PUBLIC ${CMAKE_SOURCE_DIR}/platform/bearly25/include)

#################################
# Dependencies
#################################

target_link_libraries(mm_basic PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if (PROF_COV)
  target_link_libraries(mm_basic PRIVATE gcov)
endif()
