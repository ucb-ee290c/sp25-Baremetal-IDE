#include "hal_conv.h"
#include "chip_config.h"

// --- MMIO Register Access Functions (Unchanged) ---

void reg_write8(uintptr_t addr, uint8_t data) {
volatile uint8_t *ptr = (volatile uint8_t *) addr;
*ptr = data;
}

uint8_t reg_read8(uintptr_t addr) {
volatile uint8_t *ptr = (volatile uint8_t *) addr;
return *ptr & 0xFF;
}

void reg_write16(uintptr_t addr, uint16_t data) {
volatile uint16_t *ptr = (volatile uint16_t *) addr;
*ptr = data;
}

uint16_t reg_read16(uintptr_t addr) {
volatile uint16_t *ptr = (volatile uint16_t *) addr;
return *ptr & 0xFFFF;
}

void reg_write32(uintptr_t addr, uint32_t data) {
volatile uint32_t *ptr = (volatile uint32_t *) addr;
*ptr = data;
}

uint32_t reg_read32(uintptr_t addr) {
volatile uint32_t *ptr = (volatile uint32_t *) addr;
return *ptr & 0xFFFFFFFF;
}

void reg_write64(unsigned long addr, uint64_t data) {
volatile uint64_t *ptr = (volatile uint64_t *) addr;
*ptr = data;
}

uint64_t reg_read64(unsigned long addr) {
volatile uint64_t *ptr = (volatile uint64_t *) addr;
return *ptr;
}

// =================================================================
// --- Refactored Driver Implementation ---
// Replaced &conv->MEMBER with (MMIO_BASE + MEMBER_ADDR)
// =================================================================


void conv_init() {
  // All accesses now use MMIO_BASE + OFFSET
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  
  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 1);   
  reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 1);
}

int conv_set_params(uint32_t* input, uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length){
  // Clear MMIO Reset, Clear Datapath, Stop
  reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 0);
  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 0); 
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  

  // Write input data (Streaming 64-bit writes)
  for (int i = 0; i < input_length; i += 2) {
  reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR), *((uint64_t*) (input + i)));
  }

  // Write parameters
  reg_write32((uintptr_t)(MMIO_BASE + CONV_LENGTH_ADDR), input_length);
  reg_write16((uintptr_t)(MMIO_BASE + CONV_DILATION_ADDR), dilation);

  // Write kernel data and kernel length encoding
  if (kernel_length == 8) {
  for (int i = 0; i < 8; i += 2) {
  reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
  }
  reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 0);
  } else if (kernel_length == 16) {
  for (int i = 0; i < 16; i += 2) {
  reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
  }
  reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 1);   } else {
  return -1;  }

  return 0;

}

void conv_read_output(uint32_t *output, int output_len, int *status, uint32_t* input) {
  // Read pairs of FP32s (2 per 64-bit read)
  int j; // Index for 32-bit output elements
  for (j = 0; j < output_len - 1; j += 2) {
  uint64_t current_out = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
  uint32_t *unpacked = (uint32_t *) &current_out;

  output[j]   = unpacked[0];
  output[j + 1] = unpacked[1];
  }
    
  // Handle the final odd element if output_len is odd
  if (output_len % 2 != 0) {
     uint64_t last_out = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
     uint32_t* unpacked_out = (uint32_t*) &last_out;
     output[output_len - 1] = unpacked_out[0];
  }

  // Read Status
  *status = reg_read8((uintptr_t)(MMIO_BASE + CONV_STATUS_ADDR));
}

void start_conv() {
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 1);
}

uint8_t get_register_status() {
  return reg_read8((uintptr_t)(MMIO_BASE + CONV_STATUS_ADDR));
}

uint32_t get_register_out_count() {
  return reg_read32((uintptr_t)(MMIO_BASE + CONV_COUNT_ADDR));
}

uint8_t get_register_read_check() {
  return reg_read8((uintptr_t)(MMIO_BASE + READ_CHECK_ADDR));
}
