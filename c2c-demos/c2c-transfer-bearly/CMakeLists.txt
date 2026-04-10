add_executable(c2c-transfer-bearly
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(c2c-transfer-bearly PUBLIC include)
target_include_directories(c2c-transfer-bearly PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_compile_options(c2c-transfer-bearly PRIVATE -O3)

target_link_libraries(c2c-transfer-bearly PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(c2c-transfer-bearly PRIVATE gcov)
endif()
