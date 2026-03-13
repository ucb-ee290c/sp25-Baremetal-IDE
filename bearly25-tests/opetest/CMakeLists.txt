# CMakeLists definitions for target `opetest`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(opetest
  src/main.c
  src/ope.c
)

# Header Files
target_include_directories(opetest PUBLIC include)
target_include_directories(opetest PUBLIC ${CMAKE_SOURCE_DIR}/platform/bearly25/include)

#################################
# Dependencies
#################################

target_link_libraries(opetest PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
)

if(BUILD_VECNN)
  target_link_libraries(opetest PRIVATE vecnn)
endif()

if (PROF_COV)
  target_link_libraries(opetest PRIVATE gcov)
endif()
