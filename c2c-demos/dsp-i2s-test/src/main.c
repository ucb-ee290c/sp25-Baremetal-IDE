/*
 * dsp-i2s-test — validate the DSP I2S microphone capture path.
 *
 * Standalone (no C2C link). Configures the mic like the proven dsp25-demos/nick-i2s-test, then
 * repeatedly captures a window of samples and prints statistics + a level meter + a verdict, so
 * live audio at the mic can be confirmed before wiring I2S into dsp-kws-rolling as the audio source.
 *
 * I2S read model (see platform/dsp25/include/hal_i2s.h): read_I2S_rx() pops one 64-bit block that
 * packs TWO 32-bit samples; we unpack both as consecutive time samples. A read on an empty RX FIFO
 * stalls the core until data arrives, so no busy-wait is needed.
 */

#include <stdint.h>
#include <stdbool.h>

#include "dsp_i2s_test_config.h"

#include "rocketcore.h"
#include "hal_mmio.h"

/* PLL target only documents intent; this demo intentionally does NOT reconfigure the PLL (see
 * config header) so the I2S timing matches the known-good nominal-clock setup. */
uint64_t target_frequency = 500000000;

/* Mic config: RX enabled, 32-bit samples, internal clock generator on, DAC off. Mirrors
 * nick-i2s-test's i2s_params_mic. */
static i2s_params_t g_i2s_params_mic = {
    .tx_en         = 1,
    .rx_en         = 1,
    .bitdepth_tx   = I2S_BITDEPTH_32,
    .bitdepth_rx   = I2S_BITDEPTH_32,
    .clkgen        = 1,
    .dacen         = 0,
    .ws_len        = 3,
    .clkdiv        = DSP_I2S_TEST_CLKDIV,
    .tx_fp         = 0,
    .rx_fp         = 0,
    .tx_force_left = 0,
    .rx_force_left = 0,
};

static inline int32_t abs32(int32_t x) {
  return (x < 0) ? -x : x;
}

/* Read one 64-bit RX block and unpack its two 32-bit samples (little-endian: low word first). */
static inline void read_two_samples(int32_t *s0, int32_t *s1) {
  uint64_t v = read_I2S_rx(DSP_I2S_TEST_MIC_CHANNEL, I2S_LEFT);
  *s0 = (int32_t)(uint32_t)(v & 0xFFFFFFFFu);
  *s1 = (int32_t)(uint32_t)(v >> 32);
}

/* Dump the I2S peripheral registers so we can tell a live-but-empty FIFO from a wedged one, and
 * confirm the config write took. rx_wm is the RX FIFO fill level (in 64-bit blocks): if this stays
 * 0 yet reads still return, the RX-empty status is misdecoded and reads are draining an empty FIFO
 * (which returns 0xFFFF..). If rx_wm > 0, real frames are being captured. */
/* NOTE: read the watermark registers as BYTES. The HAL getters get_I2S_rx_watermark/
 * get_I2S_tx_watermark read them as a 32-bit int at byte-offset addresses (0x102/0x103/...),
 * which is a misaligned MMIO access that HANGS this core. These registers are single bytes. */
static void print_i2s_registers(int ch) {
  uint16_t cfg = reg_read16(I2S_CONFIG(ch));
  uint8_t status = reg_read8(I2S_STATUS(ch));
  int rx_empty_l = get_I2S_rx_empty(ch, I2S_LEFT);
  uint8_t rx_wm_l = reg_read8(I2S_WATERMARK_RX_L(ch));
  uint8_t rx_wm_r = reg_read8(I2S_WATERMARK_RX_R(ch));
  uint8_t tx_wm_l = reg_read8(I2S_WATERMARK_TX_L(ch));
  DSP_I2S_TEST_LOG("[dsp-i2s-test] regs ch=%d config=0x%04x status=0x%02x rx_empty_L=%d "
                   "rx_wm(L/R)=%u/%u tx_wm_L=%u\n",
                   ch, (unsigned)cfg, (unsigned)status, rx_empty_l,
                   (unsigned)rx_wm_l, (unsigned)rx_wm_r, (unsigned)tx_wm_l);
}

static void print_startup_probe(void) {
  print_i2s_registers(DSP_I2S_TEST_CHANNEL);
  DSP_I2S_TEST_LOG("[dsp-i2s-test] raw RX probe (first %u 64-bit blocks; rx_wm before each read):\n",
                   (unsigned)DSP_I2S_TEST_STARTUP_HEX_READS);
  for (uint32_t i = 0; i < DSP_I2S_TEST_STARTUP_HEX_READS; ++i) {
    uint8_t wm = reg_read8(I2S_WATERMARK_RX_L(DSP_I2S_TEST_CHANNEL)); /* byte read (see note above) */
    int empty = get_I2S_rx_empty(DSP_I2S_TEST_CHANNEL, I2S_LEFT);
    uint64_t v = read_I2S_rx(DSP_I2S_TEST_CHANNEL, I2S_LEFT);
    DSP_I2S_TEST_LOG("  [%u] rx_wm=%d empty=%d 0x%016llx  (s0=%ld s1=%ld)\n",
                     (unsigned)i, wm, empty, (unsigned long long)v,
                     (long)(int32_t)(uint32_t)(v & 0xFFFFFFFFu),
                     (long)(int32_t)(uint32_t)(v >> 32));
  }
}

/* Capture DSP_I2S_TEST_WINDOW_READS blocks (2x samples) and report window statistics. */
static void capture_and_report(uint32_t window_idx) {
  const uint32_t nreads = DSP_I2S_TEST_WINDOW_READS;
  const uint32_t nsamp = nreads * 2u;

  int32_t s_min = INT32_MAX;
  int32_t s_max = INT32_MIN;
  int64_t sum = 0;       /* for DC/mean */
  uint64_t sum_abs = 0;  /* energy proxy (mean of |sample|) */
  uint32_t zero_cross = 0;
  uint32_t exact_zero = 0;
  int32_t prev = 0;
  int have_prev = 0;

#if DSP_I2S_TEST_RAW_DUMP
  uint32_t dump_stride = DSP_I2S_TEST_RAW_DUMP_DECIMATE;
#endif

  for (uint32_t r = 0; r < nreads; ++r) {
    int32_t a, b;
    read_two_samples(&a, &b);

    int32_t pair[2] = {a, b};
    for (int j = 0; j < 2; ++j) {
      int32_t s = pair[j];
      if (s < s_min) s_min = s;
      if (s > s_max) s_max = s;
      sum += (int64_t)s;
      sum_abs += (uint64_t)(uint32_t)abs32(s);
      if (s == 0) exact_zero++;
      if (have_prev && (((prev < 0) && (s >= 0)) || ((prev >= 0) && (s < 0)))) {
        zero_cross++;
      }
      prev = s;
      have_prev = 1;

#if DSP_I2S_TEST_RAW_DUMP
      uint32_t global_idx = (r * 2u) + (uint32_t)j;
      if ((global_idx % dump_stride) == 0u) {
        DSP_I2S_TEST_LOG("    raw[%u]=%ld\n", (unsigned)global_idx, (long)s);
      }
#endif
    }
  }

  int64_t mean = sum / (int64_t)nsamp;
  uint64_t mean_abs = sum_abs / (uint64_t)nsamp;
  int64_t pp = (int64_t)s_max - (int64_t)s_min;

  /* Scale down 32-bit magnitudes for a readable level, then classify + draw a meter. */
  uint32_t level = (uint32_t)(mean_abs >> DSP_I2S_TEST_LEVEL_SHIFT);
  const char *verdict;
  if (pp == 0) {
    verdict = "STUCK";                 /* bus not toggling at all */
  } else if (level >= DSP_I2S_TEST_SIGNAL_LEVEL) {
    verdict = "SIGNAL";
  } else if (level < DSP_I2S_TEST_SILENCE_LEVEL) {
    verdict = "SILENT";
  } else {
    verdict = "quiet";
  }

  char meter[DSP_I2S_TEST_METER_WIDTH + 1];
  uint32_t fill = (level * (uint32_t)DSP_I2S_TEST_METER_WIDTH) / DSP_I2S_TEST_METER_FULLSCALE;
  if (fill > (uint32_t)DSP_I2S_TEST_METER_WIDTH) {
    fill = (uint32_t)DSP_I2S_TEST_METER_WIDTH;
  }
  for (uint32_t i = 0; i < (uint32_t)DSP_I2S_TEST_METER_WIDTH; ++i) {
    meter[i] = (i < fill) ? '#' : '.';
  }
  meter[DSP_I2S_TEST_METER_WIDTH] = '\0';

  DSP_I2S_TEST_LOG("[dsp-i2s-test] win=%u n=%u min=%ld max=%ld pp=%lld mean=%lld absmean=%llu "
                   "lvl=%u zc=%u zeros=%u %-6s [%s]\n",
                   (unsigned)window_idx, (unsigned)nsamp,
                   (long)s_min, (long)s_max, (long long)pp,
                   (long long)mean, (unsigned long long)mean_abs,
                   (unsigned)level, (unsigned)zero_cross, (unsigned)exact_zero,
                   verdict, meter);
}

#if DSP_I2S_TEST_LOOPBACK
/* Internal loopback self-test: TX ch0 -> (external jumper GPIO18->GPIO20) -> RX ch0. Isolates the
 * chip I2S + pads + software from the mic/level-shifter. See config header for wiring. */
static void loopback_test(void) {
  DSP_I2S_TEST_LOG("[dsp-i2s-test] LOOPBACK: jumper GPIO18 (I2S_SDOUT0) -> GPIO20 (I2S_SDIN0); "
                   "DISCONNECT the mic/shifter from GPIO20 first (avoid two drivers on SDIN0).\n");

  /* This I2S master appears to clock only while TX has data (priming made RX capture; draining TX
   * stalled it). So: keep TX continuously fed to hold the clock running, and read RX ONLY when it
   * actually has data (rx_wm>0) — never force an empty-FIFO read (which returns 0xFFFF). */
  const int ch = DSP_I2S_TEST_CHANNEL;
  uint32_t tx_ctr = 0;
  uint32_t win = 0;
  while (1) {
    int32_t mn = INT32_MAX;
    int32_t mx = INT32_MIN;
    uint32_t n_real = 0;   /* reads that came from a non-empty RX FIFO */
    uint64_t last_rx = 0;

    for (uint32_t iter = 0; iter < (DSP_I2S_TEST_WINDOW_READS * 4u); ++iter) {
      /* Top up TX (both slots) so the master clock keeps running. A Knuth hash makes the
       * transmitted stream vary strongly, so a working loopback reads back non-constant data. */
      if (!get_I2S_tx_full(ch, I2S_LEFT)) {
        uint32_t s = tx_ctr * 2654435761u;
        write_I2S_tx(ch, I2S_LEFT, ((uint64_t)s << 32) | s);
        tx_ctr++;
      }
      if (!get_I2S_tx_full(ch, I2S_RIGHT)) {
        uint32_t s = tx_ctr * 2654435761u;
        write_I2S_tx(ch, I2S_RIGHT, ((uint64_t)s << 32) | s);
        tx_ctr++;
      }

      /* Read RX only when it genuinely holds data. */
      if (reg_read8(I2S_WATERMARK_RX_L(ch)) > 0) {
        uint64_t v = read_I2S_rx(ch, I2S_LEFT);
        last_rx = v;
        n_real++;
        int32_t s0 = (int32_t)(uint32_t)(v & 0xFFFFFFFFu);
        int32_t s1 = (int32_t)(uint32_t)(v >> 32);
        if (s0 < mn) mn = s0;
        if (s0 > mx) mx = s0;
        if (s1 < mn) mn = s1;
        if (s1 > mx) mx = s1;
      }
      if (reg_read8(I2S_WATERMARK_RX_R(ch)) > 0) {
        (void)read_I2S_rx(ch, I2S_RIGHT);
      }
    }

    int64_t pp = (n_real > 0u) ? ((int64_t)mx - (int64_t)mn) : 0;
    uint16_t cfg = reg_read16(I2S_CONFIG(ch));
    uint8_t status = reg_read8(I2S_STATUS(ch));
    uint8_t rx_wm = reg_read8(I2S_WATERMARK_RX_L(ch));
    DSP_I2S_TEST_LOG("[dsp-i2s-test] LOOPBACK win=%u cfg=0x%04x status=0x%02x rx_wm=%u n_real=%u "
                     "pp=%lld last_rx=0x%016llx  %s\n",
                     (unsigned)win, (unsigned)cfg, (unsigned)status, (unsigned)rx_wm,
                     (unsigned)n_real, (long long)pp, (unsigned long long)last_rx,
                     ((n_real > 0u) && (pp > 0)) ? "VARYING -> RX+TX+pads OK (fault is external mic)"
                     : (n_real > 0u) ? "RX has data but constant"
                                     : "RX never had data (clock/serializer)");
    win++;
  }
}
#endif

void app_init(void) {
  UART_InitType uart_cfg;
  uart_cfg.baudrate = 115200;
  uart_cfg.mode = UART_MODE_TX_RX;
  uart_cfg.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &uart_cfg);

  /* NOTE: PLL intentionally left at reset/nominal so I2S clkdiv matches the proven setup. */

  DSP_I2S_TEST_LOG("[dsp-i2s-test] configuring I2S mic (channel=%d clkdiv=%d bitdepth=32)\n",
                   DSP_I2S_TEST_MIC_CHANNEL, DSP_I2S_TEST_CLKDIV);
  config_I2S(DSP_I2S_TEST_CHANNEL, &g_i2s_params_mic);
  DSP_I2S_TEST_LOG("[dsp-i2s-test] I2S configured; register readback:\n");
  print_i2s_registers(DSP_I2S_TEST_CHANNEL);
}

void app_main(void) {
#if DSP_I2S_TEST_LOOPBACK
  loopback_test(); /* never returns */
#else
  print_startup_probe();

  DSP_I2S_TEST_LOG("[dsp-i2s-test] tap/speak at the mic and watch absmean/lvl/meter respond. "
                   "STUCK=bus not toggling, SILENT=noise floor, SIGNAL=audio present.\n");

  uint32_t window_idx = 0;
  while (1) {
    capture_and_report(window_idx);
    window_idx++;
  }
#endif
}

int main(void) {
  app_init();
  app_main();
  return 0;
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    __asm__ volatile("wfi");
  }
}
