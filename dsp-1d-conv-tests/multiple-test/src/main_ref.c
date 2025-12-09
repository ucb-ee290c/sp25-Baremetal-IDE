/* 
    MUST BE RUN WITH logMaxKernelSize = 3 OR IT WILL NOT WORK.
    
    The expectation for software is to zero-extend the kernel array to fit the maximum kernel size.
    In this case, where logMaxKernelSize = 3, the maximum kernel size is 8, so the kernel array must 
    be zero extended by 0 to be of size 8.
*/
// #include "float16.h"
// #include "float16.c"
// #include "../../../tests/mmio.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#define BASE_ADDR 0x08800000
#define INPUT_ADDR      0x08800000
#define OUTPUT_ADDR     0x08800020
#define KERNEL_ADDR     0x08800040
#define START_ADDR      0x0880006C
#define CLEAR_ADDR      0x0880006D
#define LENGTH_ADDR     0x08800078
#define DILATION_ADDR   0x0880007C
union Converter {
    float f;
    uint32_t u;
};
void convolution_1D(const uint32_t *arr, size_t arr_len,
                         const uint32_t *kernel, size_t kernel_len,
                         size_t dilation, float *output) {
  /*
    Full-length 1D correlation (no kernel flip).
    Output length: arr_len + (kernel_len - 1) * dilation.
    Zero-extend outside [0, arr_len).
    NOTE: If you want mathematical *convolution*, reverse the kernel
          before calling this (or index with kernel[kernel_len-1-j]).
  */
  const size_t out_len = arr_len + (kernel_len - 1) * dilation;
  const long shift = (long)(kernel_len - 1) * (long)dilation;  // left pad
  for (size_t o = 0; o < out_len; ++o) {
    float acc = 0.0f;
    const long base = (long)o - shift;
    for (size_t j = 0; j < kernel_len; ++j) {
      const long arr_index = base + (long)j * (long)dilation;
      if (arr_index >= 0 && arr_index < (long)arr_len) {
        // Reinterpret uint32_t bits as float safely
        uint32_t xb = arr[arr_index];
        uint32_t hb = kernel[j];
        float xf, hf;
        memcpy(&xf, &xb, sizeof(float));
        memcpy(&hf, &hb, sizeof(float));
        acc += xf * hf;
      }
    }
    output[o] = acc;
  }
}
/* Bit-cast helpers */
static inline float f32_from_u32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t u32_from_f32(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* Read N FP32 values from 64-bit output port (2 floats per read) */
static void read_outputs_u32(uint32_t *dst, size_t out_len) {
  const size_t pairs = (out_len + 1) / 2;
  for (size_t i = 0; i < pairs; ++i) {
    uint64_t x = reg_read64(OUTPUT_ADDR);
    uint32_t *p = (uint32_t *)&x;       // follows platform endianness; matches your earlier unpack
    dst[2*i] = p[0];
    if (2*i + 1 < out_len) dst[2*i + 1] = p[1];
  }
}
int main(void) {
  puts("Starting test");
  
  uint32_t in_arr_1[16] = {0x3F800000,0xBF800000,0x40000000,0x40400000,0xC0800000,
                           0x40A00000,0xBF800000,0x40000000,0x41100000,0x40E00000,
                           0x40400000,0x40C00000,0x40800000,0x40A00000,0x40000000,0x40A00000};
  uint16_t in_dilation_1[1] = {1};
  uint32_t in_kernel_1[8] = {0x3F800000,0x40400000,0x40400000,0x3F800000,0,0,0,0};
  uint32_t in_arr_2[16] = {0x41200000,0x41300000,0x41400000,0x41500000,0x3F800000,0x40000000,
                           0x40400000,0x40800000,0x41000000,0x40E00000,0x40C00000,0x40A00000,
                           0x40800000,0xBF800000,0xC0000000,0xC0400000};
  uint16_t in_dilation_2[1] = {1};
  uint32_t in_kernel_2[8] = {0x3F800000,0xBF800000,0x3F800000,0,0,0,0,0};
  /* --- First Convolution (full length) --- */
  puts("Starting First Convolution");
  puts("Setting values of MMIO registers");
  reg_write64(INPUT_ADDR, *((uint64_t*) in_arr_1));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_1 + 2)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_1 + 4)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_1 + 6)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_1 + 8)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_1 + 10)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_1 + 12)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_1 + 14)));
  reg_write32(LENGTH_ADDR, 16);
  reg_write16(DILATION_ADDR, in_dilation_1[0]);
  reg_write64(KERNEL_ADDR, *((uint64_t*) in_kernel_1));
  reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel_1 + 2)));
  reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel_1 + 4)));
  reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel_1 + 6)));
  puts("Starting Convolution");
  reg_write8(START_ADDR, 1);
  puts("Waiting for convolution to complete");
  /* Print inputs/kernels (hex and decimal) */
  printf("Input (FP32): ");
  for (int i = 0; i < 16; i++) printf("%#x ", in_arr_1[i]);
  printf("\nInput (decimal): ");
  for (int i = 0; i < 16; i++) printf("%g ", f32_from_u32(in_arr_1[i]));
  printf("\nKernel (FP32): ");
  for (int i = 0; i < 8; i++) printf("%#x ", in_kernel_1[i]);
  printf("\nKernel (decimal): ");
  for (int i = 0; i < 8; i++) printf("%g ", f32_from_u32(in_kernel_1[i]));
  const size_t out_len1 = 16 + (8 - 1) * in_dilation_1[0]; // 23
  uint32_t test_out_1_bits[23] = {0};
  printf("\nTest Output (FP32 binary): ");
  read_outputs_u32(test_out_1_bits, out_len1);
  for (size_t i = 0; i < out_len1; ++i) printf("0x%08x ", test_out_1_bits[i]);
  float ref_out_1[23] = {0};
  convolution_1D(in_arr_1, 16, in_kernel_1, 8, in_dilation_1[0], ref_out_1);
  printf("\nReference Output (FP32 binary): ");
  uint32_t ref_out_1_bits[23];
  for (size_t i = 0; i < out_len1; ++i) {
    ref_out_1_bits[i] = u32_from_f32(ref_out_1[i]);
    printf("0x%08X ", ref_out_1_bits[i]);
  }
  printf("\n");
  reg_write8(START_ADDR, 0);
  reg_write8(CLEAR_ADDR, 1);
  /* --- Second Convolution (full length) --- */
  puts("Starting Second Convolution");
  puts("Setting values of MMIO registers");
  reg_write64(INPUT_ADDR, *((uint64_t*) in_arr_2));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_2 + 2)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_2 + 4)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_2 + 6)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_2 + 8)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_2 + 10)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_2 + 12)));
  reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr_2 + 14)));
  reg_write32(LENGTH_ADDR, 16);
  reg_write16(DILATION_ADDR, in_dilation_2[0]);
  reg_write64(KERNEL_ADDR, *((uint64_t*) in_kernel_2));
  reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel_2 + 2)));
  reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel_2 + 4)));
  reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel_2 + 6)));
  puts("Starting Convolution");
  reg_write8(CLEAR_ADDR, 0);
  reg_write8(START_ADDR, 1);
  puts("Waiting for convolution to complete");
  printf("Input (FP32): ");
  for (int i = 0; i < 16; i++) printf("%#x ", in_arr_2[i]);
  printf("\nInput (decimal): ");
  for (int i = 0; i < 16; i++) printf("%g ", f32_from_u32(in_arr_2[i]));
  printf("\nKernel (FP32): ");
  for (int i = 0; i < 8; i++) printf("%#x ", in_kernel_2[i]);
  printf("\nKernel (decimal): ");
  for (int i = 0; i < 8; i++) printf("%g ", f32_from_u32(in_kernel_2[i]));
  const size_t out_len2 = 16 + (8 - 1) * in_dilation_2[0]; // 23
  uint32_t test_out_2_bits[23] = {0};
  printf("\nTest Output (FP32 binary): ");
  read_outputs_u32(test_out_2_bits, out_len2);
  for (size_t i = 0; i < out_len2; ++i) printf("0x%08x ", test_out_2_bits[i]);
  float ref_out_2[23] = {0};
  convolution_1D(in_arr_2, 16, in_kernel_2, 8, in_dilation_2[0], ref_out_2);
  printf("\nReference Output (FP32 binary): ");
  uint32_t ref_out_2_bits[23];
  for (size_t i = 0; i < out_len2; ++i) {
    ref_out_2_bits[i] = u32_from_f32(ref_out_2[i]);
    printf("0x%08X ", ref_out_2_bits[i]);
  }
  printf("\n");
  /* Compare full bit patterns (all 23 values) */
  if (memcmp(test_out_1_bits, ref_out_1_bits, out_len1 * sizeof(uint32_t)) == 0 &&
      memcmp(test_out_2_bits, ref_out_2_bits, out_len2 * sizeof(uint32_t)) == 0) {
    printf("[TEST PASSED]: Test Outputs match Reference Outputs.\n");
  } else {
    printf("[TEST FAILED]: 1 or more Test Outputs do not match Reference Outputs.\n");
  }
  printf("\n");
  return 0;
}