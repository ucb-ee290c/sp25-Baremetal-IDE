# CMakeLists definitions for target `mfcc-bmarks-mc`.

if(NOT TARGET mfcclib)
  message(FATAL_ERROR "mfcc-bmarks-mc requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

if (NOT TARGET threadlib)
  message(FATAL_ERROR "mfcc-bmarks-mc requires THREAD_LIB=ON (threadlib target missing).")
endif()

add_executable(mfcc-bmarks-mc
  src/main.c
  src/bench_cache.c
  src/bench_cases.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(mfcc-bmarks-mc PUBLIC include)
target_include_directories(mfcc-bmarks-mc PRIVATE
  ${CMAKE_SOURCE_DIR}/thread-lib
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_link_libraries(mfcc-bmarks-mc PRIVATE
  mfcclib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  threadlib
)

if(PROF_COV)
  target_link_libraries(mfcc-bmarks-mc PRIVATE gcov)
endif()
