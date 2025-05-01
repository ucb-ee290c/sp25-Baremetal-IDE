# Bearly25 2D Convolution Accelerator

This directory contains the hardware abstraction layer (HAL) for the Bearly25 platform's 2D convolution accelerator.

## Quick Start

```c
#include "chip_config.h"


// Simplified wrapper function
void perform_convolution(uint64_t src_addr, uint64_t dest_addr,
                        uint64_t height, uint64_t width,
                        int8_t *kernel, uint8_t kernel_size,
                        uint8_t use_relu, uint8_t stride) {
    // Get accelerator pointer
    Conv2D_Accel_Type *conv = CONV2D;
    
    // Initialize and configure
    conv2d_init(conv);
    /* It may be worthwhile to experiment calling the perform_convolution with/without the conv2d_init function. 
    * This technically sends a write to the MMIO, which might slow things down. Try using the driver with it first, 
    *then try it without to see if you get any performance improvements / maintain accuracy.
    */
    conv2d_configure(conv, src_addr, dest_addr, height, width, 
                    kernel_size, use_relu, stride);
    
    // Set kernel values
    conv2d_set_kernel(conv, kernel, kernel_size);
    
    // Start and wait for completion
    conv2d_start(conv);
    while (!conv2d_is_ready(conv)) {
        // Wait until convolution is complete
    }
}


// Example usage:
void example() {
    // Example parameters
    uint64_t src_addr = 0x10000000;
    uint64_t dest_addr = 0x20000000;
    uint64_t height = 32;
    uint64_t width = 32;
    int8_t kernel[9] = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };
    uint8_t kernel_size = 3;
    uint8_t use_relu = 1;
    uint8_t stride = 1;
    
    // Perform convolution
    perform_convolution(src_addr, dest_addr, height, width,
                       kernel, kernel_size, use_relu, stride);
}
```

## Key Features

- Supports 3x3 and 5x5 kernel sizes
- Optional ReLU activation
- Configurable stride
- Memory-mapped at base address `0x08808000`
- Hardware-accelerated 2D convolution

## Requirements

- Input/output data must be properly aligned in memory
- Kernel values must be 8-bit signed integers (int8_t)
- Kernel size must be either 3 or 5
- Stride is supported as 1 or 2 for 3x3 kernel, 1, 2, or 4 for 5x5 kernel.

## API Reference

```c
// Memory map of the 2D Convolution accelerator
typedef struct {
  __IO uint8_t  STATUS;        // 0x00: Status register (7 bits)
  __IO uint8_t  READY;         // 0x08: Ready register (1 bit)
  __IO uint64_t SRC_ADDR;      // 0x10: Source address register
  __IO uint64_t DEST_ADDR;     // 0x20: Destination address register
  __IO uint64_t INPUT_HEIGHT;  // 0x40: Input height register
  __IO uint64_t INPUT_WIDTH;   // 0x60: Input width register
  __IO uint64_t KERNEL_REG0;   // 0x70: Kernel register part 1 (8 bytes)
  __IO uint8_t  KERNEL_REG1;   // 0x78: Kernel register part 2 (1 byte)
  __IO uint64_t KERNEL_REG2;   // 0x79: Kernel register part 3 (7 bytes)
  __IO uint64_t KERNEL_REG3;   // 0x80: Kernel register part 4 (8 bytes)
  __IO uint8_t  KERNEL_REG4;   // 0x88: Kernel register part 5 (1 byte)
  __IO uint8_t  KERNEL_SIZE;   // 0x90: Kernel size register
  __IO uint8_t  USE_RELU;      // 0x98: Use ReLU register
  __IO uint8_t  STRIDE;        // 0xA0: Stride register
} Conv2D_Accel_Type;

// Individual functions (for advanced usage)
void conv2d_init(Conv2D_Accel_Type *conv);
void conv2d_configure(Conv2D_Accel_Type *conv, 
                     uint64_t src_addr, uint64_t dest_addr,
                     uint64_t height, uint64_t width,
                     uint8_t kernel_size, uint8_t use_relu,
                     uint8_t stride);
void conv2d_set_kernel(Conv2D_Accel_Type *conv, int8_t *kernel, uint8_t size);
void conv2d_start(Conv2D_Accel_Type *conv);
void conv2d_wait_complete(Conv2D_Accel_Type *conv);
uint8_t conv2d_is_ready(Conv2D_Accel_Type *conv);  // Check if convolution is complete

// High-level wrapper function
void perform_convolution(uint64_t src_addr, uint64_t dest_addr,
                        uint64_t height, uint64_t width,
                        int8_t *kernel, uint8_t kernel_size,
                        uint8_t use_relu, uint8_t stride);
                        
uint8_t conv2d_is_ready(Conv2D_Accel_Type *conv);  // Check if convolution is complete

```

## Notes

- The accelerator includes a timeout mechanism (100M cycles) to prevent hanging
- Check the STATUS register for errors after operation completion
- The READY bit indicates operation completion

For more detailed documentation, see the source code comments in `hal_2d_conv.h` and `hal_2d_conv.c`.