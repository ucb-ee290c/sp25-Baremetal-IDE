if(NOT TARGET mfcclib)
  message(FATAL_ERROR "dsp-kws requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

add_executable(dsp-kws
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(dsp-kws PUBLIC include)
target_include_directories(dsp-kws PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

target_link_libraries(dsp-kws PRIVATE
  mfcclib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(dsp-kws PRIVATE gcov)
endif()
