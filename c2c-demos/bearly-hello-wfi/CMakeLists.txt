add_executable(bearly-hello-wfi
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(bearly-hello-wfi PUBLIC include)
target_include_directories(bearly-hello-wfi PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_compile_options(bearly-hello-wfi PRIVATE -O3)

target_link_libraries(bearly-hello-wfi PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(bearly-hello-wfi PRIVATE gcov)
endif()
