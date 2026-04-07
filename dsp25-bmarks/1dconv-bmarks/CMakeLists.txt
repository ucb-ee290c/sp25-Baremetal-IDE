# CMakeLists definitions for target `1dconv-bmarks`.

add_executable(1dconv-bmarks
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(1dconv-bmarks PUBLIC include)
target_include_directories(1dconv-bmarks PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(1dconv-bmarks PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(PROF_COV)
  target_link_libraries(1dconv-bmarks PRIVATE gcov)
endif()
