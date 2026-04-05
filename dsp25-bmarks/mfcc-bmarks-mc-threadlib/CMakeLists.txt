# CMakeLists definitions for target `mfcc-bmarks-mc-threadlib`.

if(NOT TARGET mfcclib)
  message(FATAL_ERROR "mfcc-bmarks-mc-threadlib requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

if(NOT TARGET threadlib)
  message(FATAL_ERROR "mfcc-bmarks-mc-threadlib requires THREAD_LIB=ON (threadlib target missing).")
endif()

add_executable(mfcc-bmarks-mc-threadlib
  src/main.c
  src/bench_cache.c
  src/bench_cases.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(mfcc-bmarks-mc-threadlib PUBLIC include)
target_include_directories(mfcc-bmarks-mc-threadlib PRIVATE
  ${CMAKE_SOURCE_DIR}/thread-lib
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_link_libraries(mfcc-bmarks-mc-threadlib PRIVATE
  mfcclib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
  rocketcore
  threadlib
)

if(PROF_COV)
  target_link_libraries(mfcc-bmarks-mc-threadlib PRIVATE gcov)
endif()
