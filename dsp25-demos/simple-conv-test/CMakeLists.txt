# CMakeLists definitions for target `simple-conv-test`.

add_executable(simple-conv-test
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(simple-conv-test PUBLIC include)
target_include_directories(simple-conv-test PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(simple-conv-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(PROF_COV)
  target_link_libraries(simple-conv-test PRIVATE gcov)
endif()
