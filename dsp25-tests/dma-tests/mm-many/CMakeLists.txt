add_executable(dma-mm-many
  src/main.c
)

target_include_directories(dma-mm-many PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../common
  ${CMAKE_SOURCE_DIR}/platform/dsp25
  ${CMAKE_SOURCE_DIR}/platform/dsp25/include
)

target_link_libraries(dma-mm-many PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(dma-mm-many PRIVATE gcov)
endif()
