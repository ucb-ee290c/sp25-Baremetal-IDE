#include "hal_conv.h"
#include "chip_config.h"
#include <string.h>

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


 
uint8_t perform_convolution(uint64_t inputAddrValue,     
                            uint64_t outputAddrValue, 
                            uint16_t kernelAddrValue, 
                            uint32_t lengthAddrValue, 
                            uint16_t dilationAddrValue) {

    // Define the Hardware/MMIO registers

    // --- Data Ports (Read/Write) ---
    // These are used for streaming 64-bit data packets into or out of the accelerator.
    volatile uint64_t* inputDataPtr    = (volatile uint64_t*) (MMIO_BASE + 0x00); // Input (0x00, 64-bit, W)
    volatile uint64_t* outputDataPtr   = (volatile uint64_t*) (MMIO_BASE + 0x20); // Output (0x20, 64-bit, R)
    volatile uint64_t* kernelDataPtr   = (volatile uint64_t*) (MMIO_BASE + 0x40); // Kernel (0x40, 64-bit, W)

    // --- Control and Status Registers (Read/Write) ---
    // These are used to launch the operation, check completion, and handle errors.
    volatile uint8_t* statusPtr       = (volatile uint8_t*) (MMIO_BASE + 0x6A); // Status (0x6A, 8-bit, R)
    volatile uint8_t* startPtr        = (volatile uint8_t*) (MMIO_BASE + 0x6C); // Start (0x6C, 1-bit, W)
    // volatile uint8_t* clearPtr        = (volatile uint8_t*) (MMIO_BASE + 0x6D); // Clear (0x6D, 1-bit, W)
    // volatile uint8_t* mmioResetPtr    = (volatile uint8_t*) (MMIO_BASE + 0x8F); // MMIO Reset (0x8F, 1-bit, W)

    // --- Configuration Registers (Write) ---
    // These set up the parameters for the convolution.
    volatile uint32_t* lengthPtr       = (volatile uint32_t*) (MMIO_BASE + 0x78); // Length (0x78, 32-bit, W)
    volatile uint16_t* dilationPtr     = (volatile uint16_t*) (MMIO_BASE + 0x7C); // Dilation (0x7C, 16-bit, W)
    // volatile uint8_t* doubleKernelPtr = (volatile uint8_t*) (MMIO_BASE + 0x8E); // Double Kernel (0x8E, 1-bit, W)

    // --- Monitoring and Queue Registers (Read) ---
    // These are primarily for debugging or DMA/queue management.
    // volatile uint32_t* countPtr        = (volatile uint32_t*) (MMIO_BASE + 0x70); // Count (0x70, 32-bit, R)
    // volatile uint8_t* reqEnqueuePtr   = (volatile uint8_t*) (MMIO_BASE + 0x8C); // Req Enqueue (0x8C, 1-bit, R)
    // volatile uint8_t* deqOutputValidPtr = (volatile uint8_t*) (MMIO_BASE + 0x8D); // Deq Output Valid (0x8D, 1-bit, R)

    // Connect Hardware Regs with args
    *inputDataPtr   = inputAddrValue ;
    *outputDataPtr  = outputAddrValue;
    *kernelDataPtr  = kernelAddrValue;

    asm volatile("fence");

    *lengthPtr      = lengthAddrValue;
    *dilationPtr    = dilationAddrValue;

    asm volatile("fence iorw, iorw" ::: "memory");

    // THIS STARTS THE 1D Convolution
    *startPtr = (uint8_t) 1;    // Value of 1 starts the accelerator 
                            
    // Poll until the accelerator is no longer busy
    // It is not busy when the LSB, or bit index 0 if LSB=0, of statusPtr 0
    while (*statusPtr != 0);

    asm volatile("fence iorw, iorw" ::: "memory");                          
                            
    return *statusPtr;
}



// TODO: Make sure the definitions below are correctly set!
// // Defines ALL the Hardware/MMIO registers

// // --- Data Ports (Read/Write) ---
// // These are used for streaming 64-bit data packets into or out of the accelerator.
// volatile uint64_t* inputDataPtr    = (volatile uint64_t*) (MMIO_BASE + 0x00); // Input (0x00, 64-bit, W)
// volatile uint64_t* outputDataPtr   = (volatile uint64_t*) (MMIO_BASE + 0x20); // Output (0x20, 64-bit, R)
// volatile uint64_t* kernelDataPtr   = (volatile uint64_t*) (MMIO_BASE + 0x40); // Kernel (0x40, 64-bit, W)

// // --- Control and Status Registers (Read/Write) ---
// // These are used to launch the operation, check completion, and handle errors.
// volatile uint8_t* statusPtr       = (volatile uint8_t*) (MMIO_BASE + 0x6A); // Status (0x6A, 8-bit, R)
// volatile uint8_t* startPtr        = (volatile uint8_t*) (MMIO_BASE + 0x6C); // Start (0x6C, 1-bit, W)
// volatile uint8_t* clearPtr        = (volatile uint8_t*) (MMIO_BASE + 0x6D); // Clear (0x6D, 1-bit, W)
// volatile uint8_t* mmioResetPtr    = (volatile uint8_t*) (MMIO_BASE + 0x8F); // MMIO Reset (0x8F, 1-bit, W)

// // --- Configuration Registers (Write) ---
// // These set up the parameters for the convolution.
// volatile uint32_t* lengthPtr       = (volatile uint32_t*) (MMIO_BASE + 0x78); // Length (0x78, 32-bit, W)
// volatile uint16_t* dilationPtr     = (volatile uint16_t*) (MMIO_BASE + 0x7C); // Dilation (0x7C, 16-bit, W)
// volatile uint8_t* doubleKernelPtr = (volatile uint8_t*) (MMIO_BASE + 0x8E); // Double Kernel (0x8E, 1-bit, W)

// // --- Monitoring and Queue Registers (Read) ---
// // These are primarily for debugging or DMA/queue management.
// volatile uint32_t* countPtr        = (volatile uint32_t*) (MMIO_BASE + 0x70); // Count (0x70, 32-bit, R)
// volatile uint8_t* reqEnqueuePtr   = (volatile uint8_t*) (MMIO_BASE + 0x8C); // Req Enqueue (0x8C, 1-bit, R)
// volatile uint8_t* deqOutputValidPtr = (volatile uint8_t*) (MMIO_BASE + 0x8D); // Deq Output Valid (0x8D, 1-bit, R)
