#ifndef C2C_DSP_I2S_TEST_CONFIG_H
#define C2C_DSP_I2S_TEST_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "hal_i2s.h"

/* ------------------------------------------------------------------------------------------------
 * dsp-i2s-test — standalone I2S mic-capture validation for the C2C KWS pipeline.
 *
 * Goal: prove the DSP's I2S microphone path actually captures live audio, as a stepping stone to
 * feeding live audio into dsp-kws-rolling (which today streams embedded waveforms). This demo does
 * NOT use the C2C link — it configures the mic exactly like the proven dsp25-demos/nick-i2s-test
 * (nominal 50 MHz clock, no PLL reconfig, clkdiv=8 => ~44.1 kHz), captures windows of samples, and
 * prints per-window statistics + an ASCII level meter + a STUCK/SILENT/SIGNAL verdict so you can tap
 * or speak at the mic and watch the numbers respond. See /CLAUDE.md and plan 001 (I2S live audio).
 * ---------------------------------------------------------------------------------------------- */

#ifndef DSP_I2S_TEST_LOG_ENABLE
#define DSP_I2S_TEST_LOG_ENABLE 1
#endif

#if DSP_I2S_TEST_LOG_ENABLE
#define DSP_I2S_TEST_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define DSP_I2S_TEST_LOG(...) do { } while (0)
#endif

/* I2S channel under test. 0 = mic on channel 0 (matches nick-i2s-test). Change to test another
 * channel's RX/loopback (e.g. 1, whose TX/speaker path nick-i2s-test exercised). */
#ifndef DSP_I2S_TEST_CHANNEL
#define DSP_I2S_TEST_CHANNEL 0
#endif
#ifndef DSP_I2S_TEST_MIC_CHANNEL
#define DSP_I2S_TEST_MIC_CHANNEL DSP_I2S_TEST_CHANNEL
#endif

/* Clock divider for the I2S bit clock. clkdiv=8 @ 50 MHz sys clk => ~44.1 kHz sample rate (the
 * proven value from nick-i2s-test). Sample rate is not critical for this validation. */
#ifndef DSP_I2S_TEST_CLKDIV
#define DSP_I2S_TEST_CLKDIV 8
#endif

/* Keep the I2S master clock alive during mic capture by feeding dummy TX. CONFIRMED on silicon: this
 * I2S master only generates BCLK/WS while its TX FIFO has data. With RX-only (no TX writes) the clock
 * stops, the mic is never clocked, SDIN floats, and every read is 0xFFFF. With this on, mic capture
 * writes silence to TX each iteration to hold the clock running, and reads RX only when it actually
 * has data (rx_wm>0). Leave on for any real mic capture. */
#ifndef DSP_I2S_TEST_KEEP_CLOCK
#define DSP_I2S_TEST_KEEP_CLOCK 1
#endif

/* Each read_I2S_rx() pops one 64-bit block = TWO packed 32-bit samples. A "window" is this many
 * 64-bit reads; stats are computed over 2x this many samples. 4096 reads => 8192 samples
 * (~0.19 s @ 44.1 kHz), a readable cadence of one report every fraction of a second. */
#ifndef DSP_I2S_TEST_WINDOW_READS
#define DSP_I2S_TEST_WINDOW_READS 4096u
#endif

/* Width of the ASCII level meter (characters). */
#ifndef DSP_I2S_TEST_METER_WIDTH
#define DSP_I2S_TEST_METER_WIDTH 40
#endif

/* Verdict thresholds on the window's mean |sample|, expressed as a right-shift of the 32-bit sample
 * magnitude (level = mean_abs >> DSP_I2S_TEST_LEVEL_SHIFT). Tune on-chip if the mic's noise floor
 * differs. Below SILENCE => "SILENT"; at/above SIGNAL => "SIGNAL"; in between => "quiet". */
#ifndef DSP_I2S_TEST_LEVEL_SHIFT
#define DSP_I2S_TEST_LEVEL_SHIFT 16u
#endif
#ifndef DSP_I2S_TEST_SILENCE_LEVEL
#define DSP_I2S_TEST_SILENCE_LEVEL 4u
#endif
#ifndef DSP_I2S_TEST_SIGNAL_LEVEL
#define DSP_I2S_TEST_SIGNAL_LEVEL 40u
#endif
/* Full-scale of the meter, in the same (mean_abs >> shift) units. */
#ifndef DSP_I2S_TEST_METER_FULLSCALE
#define DSP_I2S_TEST_METER_FULLSCALE 2000u
#endif

/* At startup, print this many raw 64-bit reads in hex so a wedged/constant bus (all-0x0 or
 * all-0xFFFFFFFF) is obvious before the stats loop begins. */
#ifndef DSP_I2S_TEST_STARTUP_HEX_READS
#define DSP_I2S_TEST_STARTUP_HEX_READS 8u
#endif

/* Isolation test: internal I2S loopback (no mic, no level shifter). Jumper the chip's own
 * I2S_SDOUT0 (GPIO18) directly to I2S_SDIN0 (GPIO20) — both are 1.2 V, so a direct wire is safe —
 * and DISCONNECT the mic/shifter from GPIO20 first (avoid two drivers on SDIN0). The demo then
 * transmits a known ramp on channel-0 TX and reads it back on channel-0 RX. If RX varies (pp>0),
 * the chip I2S + pads + software are good and any STUCK with the mic is an external (mic/shifter/
 * sel/power) fault. If RX is still constant, the fault is chip-side (TX drive, pad routing, clkgen).
 * 0 = normal mic capture; 1 = loopback self-test. */
#ifndef DSP_I2S_TEST_LOOPBACK
#define DSP_I2S_TEST_LOOPBACK 0
#endif

/* Optional: dump a decimated waveform (every Nth sample) once per window so you can eyeball the
 * captured signal over UART. 0 = off (stats only). */
#ifndef DSP_I2S_TEST_RAW_DUMP
#define DSP_I2S_TEST_RAW_DUMP 0
#endif
#ifndef DSP_I2S_TEST_RAW_DUMP_DECIMATE
#define DSP_I2S_TEST_RAW_DUMP_DECIMATE 256u
#endif

#endif /* C2C_DSP_I2S_TEST_CONFIG_H */
