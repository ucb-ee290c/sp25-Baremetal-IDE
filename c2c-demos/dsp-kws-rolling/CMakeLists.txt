if(NOT TARGET mfcclib)
  message(FATAL_ERROR "dsp-kws-rolling requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

set(KWS_DSP_ROLLING_USE_THREADLIB_VALUE 0)
if(TARGET threadlib)
  set(KWS_DSP_ROLLING_USE_THREADLIB_VALUE 1)
endif()

add_executable(dsp-kws-rolling
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(dsp-kws-rolling PUBLIC include)
target_include_directories(dsp-kws-rolling PRIVATE
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/thread-lib
  ${CMAKE_SOURCE_DIR}/c2c-demos/dsp-kws/include
)

target_compile_definitions(dsp-kws-rolling PRIVATE
  KWS_DSP_ROLLING_USE_THREADLIB=${KWS_DSP_ROLLING_USE_THREADLIB_VALUE}
)

target_link_libraries(dsp-kws-rolling PRIVATE
  mfcclib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)
if(TARGET threadlib)
  target_link_libraries(dsp-kws-rolling PRIVATE threadlib)
endif()

if(PROF_COV)
  target_link_libraries(dsp-kws-rolling PRIVATE gcov)
endif()
