if(NOT TARGET mfcclib)
  message(FATAL_ERROR "dsp-kws requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

set(KWS_DSP_USE_THREADLIB_VALUE 0)
if(TARGET threadlib)
  set(KWS_DSP_USE_THREADLIB_VALUE 1)
endif()

set(KWS_DSP_MODE "SAFE" CACHE STRING "dsp-kws link mode: SAFE or FAST")
set_property(CACHE KWS_DSP_MODE PROPERTY STRINGS SAFE FAST)
if (KWS_DSP_MODE STREQUAL "FAST")
  set(KWS_DSP_LINK_MODE_VALUE 1)
else()
  set(KWS_DSP_LINK_MODE_VALUE 0)
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
  KWS_DSP_USE_THREADLIB=${KWS_DSP_USE_THREADLIB_VALUE}
  KWS_DSP_LINK_MODE=${KWS_DSP_LINK_MODE_VALUE}
)

target_link_libraries(dsp-kws PRIVATE
  mfcclib
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)
if(TARGET threadlib)
  target_link_libraries(dsp-kws PRIVATE threadlib)
endif()

if(PROF_COV)
  target_link_libraries(dsp-kws PRIVATE gcov)
endif()
