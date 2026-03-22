# CMakeLists definitions for target `tinyspeech-test`.

add_executable(tinyspeech-test
  src/main.c
  src/tinyspeech_model.c
  src/tensor.c
  src/misc.c
  src/modules.c
)

target_include_directories(tinyspeech-test PUBLIC include)
target_include_directories(tinyspeech-test PRIVATE ${CMAKE_SOURCE_DIR}/tinyspeech/inference/include)
target_include_directories(tinyspeech-test PUBLIC ${CMAKE_SOURCE_DIR}/platform/dsp25/include)

target_link_libraries(tinyspeech-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if (PROF_COV)
  target_link_libraries(tinyspeech-test PRIVATE gcov)
endif()
