# CMake definitions for target `dma-bmarks`.

add_executable(dma-bmarks
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(dma-bmarks PUBLIC include)
target_include_directories(dma-bmarks PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(dma-bmarks PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(PROF_COV)
  target_link_libraries(dma-bmarks PRIVATE gcov)
endif()
