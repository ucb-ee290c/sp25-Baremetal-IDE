# dsp-i2s-test: standalone I2S mic-capture validation (DSP side; no C2C link).
# Build with the DSP toolchain, e.g.:
#   make build CHIP=dsp25 TARGET=dsp-i2s-test

add_executable(dsp-i2s-test
  src/main.c
)

target_include_directories(dsp-i2s-test PUBLIC include)
target_include_directories(dsp-i2s-test PRIVATE
  ${CMAKE_SOURCE_DIR}/bmark-lib
)

# Internal loopback self-test (jumper GPIO18->GPIO20): isolates the chip I2S/pads/software from the
# external mic + level shifter. Needs a clean reconfigure to take effect.
option(DSP_I2S_TEST_LOOPBACK "Build the I2S internal loopback self-test (TX ch0 -> RX ch0)" OFF)
set(DSP_I2S_TEST_LOOPBACK_VALUE 0)
if(DSP_I2S_TEST_LOOPBACK)
  set(DSP_I2S_TEST_LOOPBACK_VALUE 1)
  message(STATUS "dsp-i2s-test: DSP_I2S_TEST_LOOPBACK=1 (internal loopback self-test)")
endif()
target_compile_definitions(dsp-i2s-test PRIVATE
  DSP_I2S_TEST_LOOPBACK=${DSP_I2S_TEST_LOOPBACK_VALUE}
)

# glossy (whole-archive) provides the dsp25 HAL, including the I2S driver — same as
# dsp25-demos/nick-i2s-test, the proven-working I2S demo this is modeled on.
target_link_libraries(dsp-i2s-test PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(dsp-i2s-test PRIVATE gcov)
endif()
