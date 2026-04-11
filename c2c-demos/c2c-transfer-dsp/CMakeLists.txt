add_executable(c2c-transfer-dsp
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(c2c-transfer-dsp PUBLIC include)
target_include_directories(c2c-transfer-dsp PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_compile_options(c2c-transfer-dsp PRIVATE -O3)

target_link_libraries(c2c-transfer-dsp PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(c2c-transfer-dsp PRIVATE gcov)
endif()
