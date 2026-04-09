#include "main.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define KWS_TEMPLATE_CASES 8u

static mfcc_driver_t g_mfcc;
static float32_t g_templates[KWS_TEMPLATE_CASES][MFCC_DRIVER_FFT_LEN];
static float32_t g_input_window[MFCC_DRIVER_FFT_LEN];

uint64_t target_frequency = KWS_DSP_TARGET_FREQUENCY_HZ;

static float32_t clampf_local(float32_t x, float32_t lo, float32_t hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static void prepare_templates(void) {
  uint32_t lcg = 0x12345678u;
  const float32_t kPi = (float32_t)M_PI;

  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; ++n) {
    const float32_t t = (float32_t)n / MFCC_DRIVER_SAMPLE_RATE_HZ;
    const float32_t frac = (float32_t)n / (float32_t)(MFCC_DRIVER_FFT_LEN - 1u);

    g_templates[0][n] = 0.0f;
    g_templates[1][n] = (n == 0u) ? 1.0f : 0.0f;
    g_templates[2][n] = (n & 1u) ? -0.9f : 0.9f;
    g_templates[3][n] = 0.9f * sinf(2.0f * kPi * 440.0f * t);
    g_templates[4][n] = 0.9f * sinf(2.0f * kPi * 3000.0f * t);

    {
      const float32_t chirp_hz = 100.0f + (2900.0f * frac);
      g_templates[5][n] = 0.85f * sinf(2.0f * kPi * chirp_hz * t);
    }

    lcg = (1664525u * lcg) + 1013904223u;
    {
      const float32_t u = ((float32_t)(lcg & 0x00FFFFFFu) / 8388607.5f) - 1.0f;
      g_templates[6][n] = 0.8f * u;
    }

    {
      const float32_t mix =
          1.4f * sinf(2.0f * kPi * 700.0f * t) + 1.1f * sinf(2.0f * kPi * 1200.0f * t);
      g_templates[7][n] = clampf_local(mix, -0.7f, 0.7f);
    }
  }
}

static void build_window(uint16_t case_id, uint8_t frame_idx, float32_t *dst) {
  const uint32_t template_idx = ((uint32_t)case_id + (uint32_t)frame_idx) % KWS_TEMPLATE_CASES;
  const uint32_t shift = (((uint32_t)case_id * 17u) + ((uint32_t)frame_idx * 11u)) % MFCC_DRIVER_FFT_LEN;
  const float32_t *src = g_templates[template_idx];

  for (uint32_t i = 0; i < MFCC_DRIVER_FFT_LEN; ++i) {
    dst[i] = src[(i + shift) % MFCC_DRIVER_FFT_LEN];
  }
}

static int8_t quantize_mfcc(float32_t x) {
  float32_t qf = (x * (float32_t)KWS_DSP_MFCC_QUANT_SCALE) + (float32_t)KWS_DSP_MFCC_QUANT_ZERO;
  int32_t qi = (int32_t)lrintf(qf);
  if (qi > 127) {
    qi = 127;
  }
  if (qi < -127) {
    qi = -127;
  }
  return (int8_t)qi;
}

static void init_tx_uart(void) {
  UART_InitType uart_cfg;
  uart_cfg.baudrate = KWS_DSP_UART_BAUDRATE;
  uart_cfg.mode = UART_MODE_TX_RX;
  uart_cfg.stopbits = UART_STOPBITS_2;
  uart_init(KWS_DSP_TX_UART, &uart_cfg);
  KWS_DSP_TX_UART->DIV = (uint32_t)((target_frequency / KWS_DSP_UART_BAUDRATE) - 1u);
}

static void uart_send_bytes(const uint8_t *bytes, uint16_t len) {
  (void)uart_transmit(KWS_DSP_TX_UART, bytes, len, 0u);
}

static void send_packet(uint8_t type,
                        uint32_t seq,
                        uint16_t case_id,
                        uint8_t frame_idx,
                        const int8_t *payload,
                        uint8_t payload_len) {
  uint8_t packet[KWS_WIRE_MAX_PACKET_BYTES];
  const uint16_t wire_len = (uint16_t)(KWS_WIRE_FIXED_HEADER_BYTES + payload_len + KWS_WIRE_CRC_BYTES);
  uint32_t crc;

  packet[0] = KWS_SYNC0;
  packet[1] = KWS_SYNC1;
  packet[2] = KWS_PROTO_VERSION;
  packet[3] = type;
  kws_write_u32_le(&packet[4], seq);
  kws_write_u16_le(&packet[8], case_id);
  packet[10] = frame_idx;
  packet[11] = payload_len;

  if ((payload != NULL) && (payload_len > 0u)) {
    memcpy(&packet[12], payload, payload_len);
  }

  crc = kws_crc32(&packet[2], (size_t)(10u + payload_len));
  kws_write_u32_le(&packet[12u + payload_len], crc);

  uart_send_bytes(packet, wire_len);
}

static void send_case_stream(uint16_t case_id, uint32_t *seq_io, uint64_t *mfcc_cycle_sum_io) {
  uint32_t seq = *seq_io;
  float32_t mfcc_f32[MFCC_DRIVER_NUM_DCT];
  int8_t mfcc_q[KWS_MFCC_DIM];

  send_packet(KWS_PKT_CASE_START, seq++, case_id, 0xFFu, NULL, 0u);

  for (uint8_t frame_idx = 0; frame_idx < (uint8_t)KWS_DSP_FRAMES_PER_CASE; ++frame_idx) {
    uint64_t mfcc_cycles = 0;
    mfcc_driver_status_t st;

    build_window(case_id, frame_idx, g_input_window);

    st = mfcc_driver_run_sp1024x23x12_f32(&g_mfcc, g_input_window, mfcc_f32, &mfcc_cycles);
    if (st != MFCC_DRIVER_OK) {
      st = mfcc_driver_run_f32(&g_mfcc, g_input_window, mfcc_f32, &mfcc_cycles);
    }

    if (st != MFCC_DRIVER_OK) {
      KWS_DSP_LOG("[dsp-kws] MFCC failed case=%u frame=%u err=%s\n",
                  (unsigned)case_id,
                  (unsigned)frame_idx,
                  mfcc_driver_status_str(st));
      continue;
    }

    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      mfcc_q[k] = quantize_mfcc(mfcc_f32[k]);
    }

    send_packet(KWS_PKT_FRAME, seq++, case_id, frame_idx, mfcc_q, (uint8_t)KWS_MFCC_DIM);
    *mfcc_cycle_sum_io += mfcc_cycles;
  }

  send_packet(KWS_PKT_CASE_END, seq++, case_id, 0xFFu, NULL, 0u);
  *seq_io = seq;
}

void app_init(void) {
  init_test(target_frequency);
  init_tx_uart();

  prepare_templates();
  if (mfcc_driver_init(&g_mfcc) != MFCC_DRIVER_OK) {
    KWS_DSP_LOG("[dsp-kws] MFCC init failed\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

  KWS_DSP_LOG("[dsp-kws] UART stream init: uart=%p baud=%u\n",
              (void *)KWS_DSP_TX_UART,
              (unsigned)KWS_DSP_UART_BAUDRATE);
}

void app_main(void) {
  uint32_t seq = 1u;
  uint64_t total_mfcc_cycles = 0u;

  for (uint16_t case_id = 0; case_id < (uint16_t)KWS_DSP_NUM_CASES; ++case_id) {
    send_case_stream(case_id, &seq, &total_mfcc_cycles);

    if (KWS_DSP_LOG_ENABLE && ((case_id + 1u) % KWS_DSP_PROGRESS_EVERY_CASES == 0u)) {
      KWS_DSP_LOG("[dsp-kws] streamed %u/%u cases\n",
                  (unsigned)(case_id + 1u),
                  (unsigned)KWS_DSP_NUM_CASES);
    }
  }

  send_packet(KWS_PKT_STREAM_END, seq++, 0xFFFFu, 0xFFu, NULL, 0u);

  KWS_DSP_LOG("[dsp-kws] done: cases=%u frames/case=%u seq_end=%u mfcc_cycles=%llu\n",
              (unsigned)KWS_DSP_NUM_CASES,
              (unsigned)KWS_DSP_FRAMES_PER_CASE,
              (unsigned)(seq - 1u),
              (unsigned long long)total_mfcc_cycles);

  while (1) {
    __asm__ volatile("wfi");
  }
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
