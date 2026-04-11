add_executable(dsp-simpletest
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(dsp-simpletest PUBLIC include)
target_include_directories(dsp-simpletest PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_compile_options(dsp-simpletest PRIVATE -O3)

target_link_libraries(dsp-simpletest PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(dsp-simpletest PRIVATE gcov)
endif()
