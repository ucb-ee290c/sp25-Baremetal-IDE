# CMakeLists definitions for target `acc-conv`.

#################################
# Build Configuration
#################################

add_executable(acc-conv
  src/main.c
  src/bench_impl.c
  src/bench_sizes.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(acc-conv PUBLIC include)
target_include_directories(acc-conv PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

#################################
# Dependencies
#################################

target_link_libraries(acc-conv PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
target_link_options(acc-conv PRIVATE -Wl,--defsym=__heap_size=16777216)

if (PROF_COV)
  target_link_libraries(acc-conv PRIVATE gcov)
endif()
