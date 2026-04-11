add_executable(bearly-simpletest
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(bearly-simpletest PUBLIC include)
target_include_directories(bearly-simpletest PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_compile_options(bearly-simpletest PRIVATE -O3)

target_link_libraries(bearly-simpletest PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(bearly-simpletest PRIVATE gcov)
endif()
