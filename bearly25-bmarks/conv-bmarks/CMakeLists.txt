# CMakeLists definitions for target `conv-bmarks`.

#################################
# Build Configuration
#################################

add_executable(conv-bmarks
  src/main.c
  src/conv_bench.c
  src/conv_cases.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
  ${CMAKE_SOURCE_DIR}/platform/bearly25/src/hal_2d_conv.c
)

target_include_directories(conv-bmarks PUBLIC include)
target_include_directories(conv-bmarks PRIVATE
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/platform/bearly25
  ${CMAKE_SOURCE_DIR}/platform/bearly25/include
)

#################################
# Dependencies
#################################

target_link_libraries(conv-bmarks PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(conv-bmarks PRIVATE gcov)
endif()
