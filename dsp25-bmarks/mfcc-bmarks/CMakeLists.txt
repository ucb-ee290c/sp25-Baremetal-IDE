# CMakeLists definitions for target `mfcc-bmarks`.

if(NOT TARGET mfcclib)
  message(FATAL_ERROR "mfcc-bmarks requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

add_executable(mfcc-bmarks
  src/main.c
  src/bench_cache.c
  src/bench_cases.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(mfcc-bmarks PUBLIC include)
target_include_directories(mfcc-bmarks PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(mfcc-bmarks PRIVATE
  mfcclib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(PROF_COV)
  target_link_libraries(mfcc-bmarks PRIVATE gcov)
endif()
