add_executable(dma-mm-memcpy-bench
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(dma-mm-memcpy-bench PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../common
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/platform/dsp25
  ${CMAKE_SOURCE_DIR}/platform/dsp25/include
)

target_link_libraries(dma-mm-memcpy-bench PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(dma-mm-memcpy-bench PRIVATE gcov)
endif()
