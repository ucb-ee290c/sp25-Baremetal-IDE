# CMakeLists definitions for target `convtest`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(convtest
  src/main.c
)

# Header Files
target_include_directories(convtest PUBLIC include)
target_include_directories(convtest PUBLIC ${CMAKE_SOURCE_DIR}/platform/bearly25/include)

#################################
# Dependencies
#################################

target_link_libraries(convtest PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if(BUILD_VECNN)
  target_link_libraries(convtest PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(convtest PRIVATE gcov)
endif()
