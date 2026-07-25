if(NOT TARGET mfcclib)
  message(FATAL_ERROR "dsp-kws-rolling requires target `mfcclib`. Configure with -DBUILD_MFCC_LIB=ON.")
endif()

set(KWS_DSP_ROLLING_USE_THREADLIB_VALUE 0)
if(TARGET threadlib)
  set(KWS_DSP_ROLLING_USE_THREADLIB_VALUE 1)
endif()

# Multi-testcase mode (plan 003 axis B): round-robin the generated waveform set
# (include/kws_dsp_signals.h) instead of the single embedded yes_test_005. Generate the header
# first with scripts/gen_dsp_signals.py. Needs a clean reconfigure to take effect.
option(KWS_DSP_ROLLING_MULTI_SIGNAL "Stream generated multi-keyword waveforms (needs include/kws_dsp_signals.h)" OFF)
set(KWS_DSP_ROLLING_MULTI_SIGNAL_VALUE 0)
if(KWS_DSP_ROLLING_MULTI_SIGNAL)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/kws_dsp_signals.h")
    message(FATAL_ERROR
      "KWS_DSP_ROLLING_MULTI_SIGNAL=ON but include/kws_dsp_signals.h is missing. "
      "Generate it: python c2c-demos/dsp-kws-rolling/scripts/gen_dsp_signals.py")
  endif()
  set(KWS_DSP_ROLLING_MULTI_SIGNAL_VALUE 1)
  message(STATUS "dsp-kws-rolling: KWS_DSP_ROLLING_MULTI_SIGNAL=1 (multi-keyword waveform stream)")
endif()

# Live I2S microphone audio source (plan 001 P1). Captures ~1 s of live audio from the DSP I2S mic
# (proven in dsp-i2s-test) and runs the same MFCC -> stream pipeline on it instead of an embedded
# waveform. Mutually exclusive with KWS_DSP_ROLLING_MULTI_SIGNAL. Needs a clean reconfigure.
option(KWS_DSP_ROLLING_USE_MIC "Use live I2S mic capture as the audio source" OFF)
set(KWS_DSP_ROLLING_USE_MIC_VALUE 0)
if(KWS_DSP_ROLLING_USE_MIC)
  if(KWS_DSP_ROLLING_MULTI_SIGNAL)
    message(FATAL_ERROR "KWS_DSP_ROLLING_USE_MIC and KWS_DSP_ROLLING_MULTI_SIGNAL are mutually exclusive.")
  endif()
  set(KWS_DSP_ROLLING_USE_MIC_VALUE 1)
  message(STATUS "dsp-kws-rolling: KWS_DSP_ROLLING_USE_MIC=1 (live I2S mic audio source)")
endif()

add_executable(dsp-kws-rolling
  src/main.c
  ${CMAKE_SOURCE_DIR}/c2c-demos/common/c2c_shm.c
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
  KWS_DSP_ROLLING_MULTI_SIGNAL=${KWS_DSP_ROLLING_MULTI_SIGNAL_VALUE}
  KWS_DSP_ROLLING_USE_MIC=${KWS_DSP_ROLLING_USE_MIC_VALUE}
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
