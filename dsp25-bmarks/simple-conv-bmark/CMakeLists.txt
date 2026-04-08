# CMakeLists definitions for target `simple-conv-bmark`.

add_executable(simple-conv-bmark
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(simple-conv-bmark PUBLIC include)
target_include_directories(simple-conv-bmark PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(simple-conv-bmark PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(PROF_COV)
  target_link_libraries(simple-conv-bmark PRIVATE gcov)
endif()
