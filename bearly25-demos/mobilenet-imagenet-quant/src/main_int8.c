/*
 * main_int8.c - Mailbox-driven INT8 MobileNetV2 inference
 *
 * Host workflow:
 *  1) Write input.bin (float32 NCHW 1x3x224x224, 0x93000 bytes) into DRAM at img_addr
 *  2) Write mailbox fields at MAILBOX_ADDR (0x8F000000) in DRAM
 *  3) Set g_mbox->status = READY
 *
 * Firmware polls g_mbox, runs entry(), writes result_top1, sets status DONE.
 */

#include <stdint.h>
#include <stdio.h>
#include "model_int8.c"
#include "imagenet_labels.h"
#include "simple_setup.h"

// Hardcoded mailbox address in scratchpad memory (uncached, avoids coherency issues)
// Scratchpad: 0x08000000-0x0800FFFF (64KB total), mailbox at 4KB offset
#define MAILBOX_ADDR 0x08001000UL

#define MBOX_MAGIC 0x4D424F58u

typedef enum {
  MBOX_IDLE  = 0,
  MBOX_READY = 1,
  MBOX_BUSY = 2,
  MBOX_DONE = 3,
  MBOX_ERR = 4,
} mbox_status_t;

uint64_t target_frequency = 500000000l;

#define IMG_FORMAT_F32_NCHW_1x3x224x224 1u
#define IMG_BYTES_F32_1x3x224x224 (1u * 3u * 224u * 224u * 4u)  // 602112 = 0x93000

typedef struct __attribute__((packed, aligned(64))) {
  volatile uint32_t magic;      // 0x00
  volatile uint32_t status;     // 0x04
  volatile uint32_t seq;        // 0x08
  volatile uint32_t img_bytes;  // 0x0C
  volatile uint64_t img_addr;   // 0x10 (low @ 0x10, high @ 0x14)
  volatile uint32_t img_format; // 0x18
  volatile uint32_t reserved0;  // 0x1C
  volatile uint32_t result_top1;// 0x20
  volatile uint32_t err_code;   // 0x24
} mailbox_t;

// Pointer to mailbox at fixed DRAM address (no linker section needed)
#define g_mbox (*(volatile mailbox_t*)MAILBOX_ADDR)

static inline void fence_rw(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static int argmax(const float* arr, int n) {
  int max_idx = 0;
  float max_val = arr[0];
  for (int i = 1; i < n; i++) {
    if (arr[i] > max_val) {
      max_val = arr[i];
      max_idx = i;
    }
  }
  return max_idx;
}

static void tiny_delay(volatile uint32_t iters) {
  while (iters--) { asm volatile("nop"); }
}

int main(void) {
  init_test(target_frequency);

  printf("==============================================\n");
  printf("MobileNetV2 INT8 Inference\n");
  printf("==============================================\n");

  printf("Expect host to provide:\n");
  printf("  - input tensor: float32 [1,3,224,224] (NCHW), %u bytes (0x%x)\n",
         (unsigned)IMG_BYTES_F32_1x3x224x224, (unsigned)IMG_BYTES_F32_1x3x224x224);
  printf("  - mailbox at scratchpad address 0x%lx (uncached)\n\n", (unsigned long)MAILBOX_ADDR);

  // Initialize mailbox
  g_mbox.magic = MBOX_MAGIC;
  g_mbox.status = MBOX_IDLE;
  g_mbox.seq = 0;
  g_mbox.img_bytes = 0;
  g_mbox.img_addr = 0;
  g_mbox.img_format = 0;
  g_mbox.result_top1 = 0;
  g_mbox.err_code = 0;
  fence_rw();

  uint32_t last_seq = 0;

  while (1) {
    // Debug: print status periodically
    static uint32_t debug_counter = 0;
    if (debug_counter++ % 1000000 == 0) {  // Print every ~1M iterations
      printf("Polling... status=%u seq=%u last_seq=%u\n", 
             (unsigned)g_mbox.status, (unsigned)g_mbox.seq, (unsigned)last_seq);
    }
    
    // Wait for READY with a new seq
    // Read status first, then fence, then read other fields
    if (g_mbox.status != MBOX_READY || g_mbox.seq == last_seq) {
      tiny_delay(5000);
      continue;
    }

    // Fence to ensure we see all mailbox fields written by host
    fence_rw();

    // atomically transition from READY -> BUSY
    g_mbox.status = MBOX_BUSY;
    fence_rw();

    // Capture fields locally to avoid repeated volatile reads
    uint32_t magic = g_mbox.magic;
    uint32_t img_format = g_mbox.img_format;
    uint32_t img_bytes = g_mbox.img_bytes;
    uint64_t img_addr = g_mbox.img_addr;
    uint32_t seq = g_mbox.seq;

    // Validate request
    if (magic != MBOX_MAGIC) {
      g_mbox.err_code = 1; // bad magic
      g_mbox.status = MBOX_ERR;
      fence_rw();
      continue;
    }
    if (img_format != IMG_FORMAT_F32_NCHW_1x3x224x224) {
      g_mbox.err_code = 2; // bad format
      g_mbox.status = MBOX_ERR;
      fence_rw();
      continue;
    }
    if (img_bytes != IMG_BYTES_F32_1x3x224x224) {
      g_mbox.err_code = 3; // bad size
      g_mbox.status = MBOX_ERR;
      fence_rw();
      continue;
    }
    if (img_addr == 0) {
      g_mbox.err_code = 4; // null addr
      g_mbox.status = MBOX_ERR;
      fence_rw();
      continue;
    }

    // Cast the image address to the input tensor type
    const float (*input)[3][224][224] = (const float (*)[3][224][224]) (uintptr_t)img_addr;

    // Output logits (static to avoid stack allocation of 4KB)
    static float output[1][1000];

    printf("\n[SEQ %u] Running inference. img_addr=0x%08x%08x\n",
           (unsigned)seq,
           (unsigned)((uint64_t)img_addr >> 32),
           (unsigned)((uint64_t)img_addr & 0xffffffffu));

    // Run inference
    entry(input[0], output);

    // Result
    int predicted = argmax(output[0], 1000);
    g_mbox.result_top1 = (uint32_t)predicted;
    g_mbox.err_code = 0;

    printf("[SEQ %u] Predicted: %d (%s)\n",
           (unsigned)seq,
           predicted,
           imagenet_labels[predicted]);

    // Publish DONE and update last_seq
    fence_rw();
    last_seq = seq;
    g_mbox.status = MBOX_DONE;
    fence_rw();
    
    // Note: Status stays DONE until host writes READY again with new seq.
    // The host is responsible for the DONE -> READY transition.
  }

  return 0;
}
