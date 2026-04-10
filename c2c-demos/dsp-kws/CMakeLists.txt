if(NOT TARGET mfcclib)
  message(FATAL_ERROR "dsp-kws requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

if(NOT TARGET threadlib)
  message(FATAL_ERROR "dsp-kws requires THREAD_LIB=ON (threadlib target missing).")
endif()

add_executable(dsp-kws
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(dsp-kws PUBLIC include)
target_include_directories(dsp-kws PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/thread-lib
)

target_compile_definitions(dsp-kws PRIVATE
  KWS_DSP_USE_THREADLIB=1
)

target_link_libraries(dsp-kws PRIVATE
  mfcclib
  threadlib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(dsp-kws PRIVATE gcov)
endif()
