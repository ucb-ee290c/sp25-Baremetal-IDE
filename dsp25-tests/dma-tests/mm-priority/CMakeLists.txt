add_executable(dma-mm-priority
  src/main.c
)

target_include_directories(dma-mm-priority PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../common
  ${CMAKE_SOURCE_DIR}/platform/dsp25
  ${CMAKE_SOURCE_DIR}/platform/dsp25/include
)

target_link_libraries(dma-mm-priority PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(dma-mm-priority PRIVATE gcov)
endif()
