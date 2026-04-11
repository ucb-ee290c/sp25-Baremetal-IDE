/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/

/* Benchmarks to test the memcpy/sbus bandwidth of the chip using naive CPU code, glibc, rvv, and the DSP24 DMA Engine
   Expects a 32 bit int payload for the random seed for each iteration and returns a packet containing 1 64 bit unsigned
   integer which is the number of cycles used and a byte which indicates whether or not the copy occured correctly.
*/
#include "main.h"
#include "libbmark.h"
#include <riscv_vector.h>
#include "mtwister.h"

uint32_t TEST_SIZE = 0x200000;
volatile void* srcbuf;
volatile void* dstbuf;

typedef struct {
  uint64_t cycles;
  bool correct;
} memcpy_result_t;

static uint64_t throughput_x100(uint32_t bytes, uint64_t cycles, uint64_t bytes_per_unit) {
  uint64_t freq_hz;
  uint64_t numer;
  uint64_t denom;

  if (cycles == 0u || bytes_per_unit == 0u || chip_freq <= 0) {
    return 0u;
  }

  freq_hz = (uint64_t)chip_freq;
  numer = (uint64_t)bytes * freq_hz * 100u;
  denom = cycles * bytes_per_unit;

  if (denom == 0u) {
    return 0u;
  }

  return numer / denom;
}

static void log_bandwidth_result(const char *label, const memcpy_result_t *result) {
  uint64_t mbps_x100 = throughput_x100(TEST_SIZE, result->cycles, 1000000u);
  uint64_t mibps_x100 = throughput_x100(TEST_SIZE, result->cycles, 1048576u);

  printf("[%s] cycles=%llu bw=%llu.%02llu MB/s (%llu.%02llu MiB/s) pass=%d\n",
         label,
         (unsigned long long)result->cycles,
         (unsigned long long)(mbps_x100 / 100u),
         (unsigned long long)(mbps_x100 % 100u),
         (unsigned long long)(mibps_x100 / 100u),
         (unsigned long long)(mibps_x100 % 100u),
         result->correct ? 1 : 0);
}

void init_buffer(volatile uint32_t* buf, uint32_t size, uint32_t seed) {
  MTRand r = seedRand(seed);
  for (int i = 0; i < size/8; i++) {
    buf[i] = genRandLong(&r);
  }
}

bool check_buffer(volatile uint32_t* buf, uint32_t size, uint32_t seed) {
  MTRand r = seedRand(seed);
  for (int i = 0; i < size/8; i++) {
    if(buf[i] != genRandLong(&r)) {
      return false;
    }
  }
  return true;
}

void touch_buffer(volatile uint8_t* buf, int size) {
  volatile uint8_t var;
  for (int i = 0; i < size; i += 16) {
    var = buf[i];
  }
}

void func_test(int seed) {
  memcpy_result_t result;
  uint64_t time;
  init_buffer(srcbuf, TEST_SIZE, seed);

  start_roi();
  result.cycles = 431987423;
  end_roi();
  result.correct = check_buffer(srcbuf, TEST_SIZE, seed);
  xmit_payload_packet(&result, 9);

}

void cpu_memcpy(int seed) {
  memcpy_result_t result;
  uint64_t time;
  init_buffer(srcbuf, TEST_SIZE, seed);
  touch_buffer(dstbuf, TEST_SIZE);

  start_roi();
  time = get_cycles();
  volatile uint64_t* src = srcbuf;
  volatile uint64_t* dst = dstbuf;
  
  for (int i = 0; i < TEST_SIZE/8; i++) {
    dst[i] = src[i];
  }
  result.cycles = get_cycles() - time;
  end_roi();
  result.correct = check_buffer(dstbuf, TEST_SIZE, seed);
  log_bandwidth_result("cpu_memcpy", &result);
  xmit_payload_packet(&result, 9);
}

void glibc_memcpy(int seed) {
  memcpy_result_t result;
  uint64_t time;
  init_buffer(srcbuf, TEST_SIZE, seed);

  start_roi();
  time = get_cycles();

  memcpy(dstbuf, srcbuf, TEST_SIZE);
  result.cycles = get_cycles() - time;
  end_roi();
  result.correct = check_buffer(dstbuf, TEST_SIZE, seed);
  log_bandwidth_result("glibc_memcpy", &result);
  xmit_payload_packet(&result, 9);
}

void rvv_memcpy(int seed) {
  memcpy_result_t result;
  uint64_t time;

  volatile void* srcbuf_ptr;
  volatile void* dstbuf_ptr;
  uint32_t remaining = TEST_SIZE;

  init_buffer(srcbuf, TEST_SIZE, seed);

  start_roi();
  time = get_cycles();

  srcbuf_ptr = srcbuf;
  dstbuf_ptr = dstbuf;
  for (size_t vl; remaining > 0; remaining -= vl, srcbuf_ptr += vl, dstbuf_ptr += vl) {
    vl = __riscv_vsetvl_e8m8(TEST_SIZE);

    vuint8m8_t vec_src = __riscv_vle8_v_u8m8(srcbuf_ptr, vl);
    __riscv_vse8_v_u8m8(dstbuf_ptr, vec_src, vl);
  }
  
  result.cycles = get_cycles() - time;
  end_roi();
  result.correct = check_buffer(dstbuf, TEST_SIZE, seed);
  log_bandwidth_result("rvv_memcpy", &result);
  xmit_payload_packet(&result, 9);
}

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  srcbuf = aligned_alloc(64, TEST_SIZE);
  dstbuf = aligned_alloc(64, TEST_SIZE);
  while (1) {
    test_info t = init_test(UART1);
    int seed = *((int*) &t.payload);
    switch (t.testid) {
      case 0:
        cpu_memcpy(seed);
        break;
      case 1:
        glibc_memcpy(seed);
        break;
      case 2:
        rvv_memcpy(seed);
        break;
      default:
        func_test(seed);
        break;
    }
    clean_test(t);
  }


  /* USER CODE END WHILE */
}

/*
 * Main function for secondary harts
 * 
 * Multi-threaded programs should provide their own implementation.
 */
void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
   asm volatile ("wfi");
  }
}
