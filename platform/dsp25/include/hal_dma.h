/**
 * \file    hal_dma.h
 * \brief   DSP DMA driver.
 * \version 0.1
 * 
 * \copyright Copyright (c) 2025
 * 
 */

 #ifndef __HAL_DMA_H__
 #define __HAL_DMA_H__
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 // Baremetal IDE Definitions //
 #include "metal.h"
 #include <stdbool.h>
 #include <stdint.h>
 #include <riscv-pk/encoding.h>
 
// // Base registers
// #define DMA_MMIO_BASE 0x8812000
// #define DMA_RESET DMA_MMIO_BASE
// #define DMA_INFLIGHT_STATUS DMA_MMIO_BASE + 0x1

// // Helper constants 
// #define CHANNEL_BASE DMA_MMIO_BASE + 0x100
// #define CHANNEL_OFFSET 64
// #define INTERRUPT_BASE DMA_MMIO_BASE + 16
// #define INTERRUPT_OFFSET 16

// #define DMA_INT_CORE_0_OFFSET INTERRUPT_OFFSET * 0 // core 0
// #define DMA_INT_CORE_1_OFFSET INTERRUPT_OFFSET * 1 // core 1

// // Per core registers
// #define DMA_INT_SERVICED INTERRUPT_BASE + 0x00
// #define DMA_INT_VALID INTERRUPT_BASE + 0x01
// #define DMA_INT_TRANSACTION_ID INTERRUPT_BASE + 0x02
// #define DMA_INT_IS_ERROR INTERRUPT_BASE + 0x04
// #define DMA_INT_ADDRESS INTERRUPT_BASE + 0x08

// // Per channel registers
// #define DMA_START CHANNEL_BASE
// #define DMA_BUSY DMA_START + 0x28
// #define DMA_READY DMA_START + 0x2
// #define DMA_FIFO_LENGTH DMA_START + 0x3
// #define DMA_CORE_ID DMA_START + 0x4
// #define DMA_TRANSACTION_ID DMA_START + 0x8
// #define DMA_PERIPHERAL_ID DMA_START + 0xA
// #define DMA_TRANSACTION_PRIORITY DMA_START + 0xC
// #define DMA_MODE DMA_START + 0xE
// #define DMA_ADDR_R DMA_START + 0x10
// #define DMA_ADDR_W DMA_START + 0x18
// #define DMA_LEN DMA_START + 0x20
// #define DMA_LOGW DMA_START + 0x22
// #define DMA_INC_R DMA_START + 0x24
// #define DMA_INC_W DMA_START + 0x26

// // interrupt-specific constants
// #define DMA_INTERRUPT_ID_CORE_0 4
// #define DMA_INTERRUPT_ID_CORE_1 5

// // PLIC MMIO
// #define PLIC_BASE 0x0C000000
// #define PLIC_ENABLE_CORE_0 PLIC_BASE + 0x2000
// #define PLIC_ENABLE_CORE_1 PLIC_BASE + 0x2080

// #define PLIC_PRIORITY_DMA_INT_SRC_1 PLIC_BASE + (4 * DMA_INTERRUPT_ID_CORE_0)
// #define PLIC_PRIORITY_DMA_INT_SRC_2 PLIC_BASE + (4 * DMA_INTERRUPT_ID_CORE_1)

// #define PLIC_CLAIM_CORE_0 PLIC_BASE + 0x200004 
// #define PLIC_CLAIM_CORE_1 PLIC_BASE + 0x201004

  typedef struct {
    // ==========================================
    // Global DMA MMIO Registers
    // Base Address: 0x08812000
    // ==========================================
    __IO uint8_t DMA_RESET;         // 0x00: DMA Reset Signal. RW
    __I  uint8_t INFLIGHT_STATUS;   // 0x01: Number of current inflight transactions. R

    // ==========================================
    // Per-Core Interrupt MMIO Registers
    // Base Offset: 0x10 + (core_id * 0x10)
    // ==========================================
    // Max 9 cores by definition of MMIO (since range is 0x10 to 0x100, with 0x10 per-core stride).
    __O  uint8_t  INTERRUPT_SERVICED;         // 0x00: An ACK, inform DMA interrupt received. W
    __I  uint8_t  INTERRUPT_VALID;            // 0x01: Pending interrupt exists. R
    __I  uint16_t INTERRUPT_TRANSACTION_ID;   // 0x02: Transaction ID causing interrupt. R
    __I  uint8_t  INTERRUPT_IS_ERROR;         // 0x04: Interrupting transaction errored. R
    __I  uint64_t INTERRUPT_ADDRESS;          // 0x08: Interrupting transaction errored while reading/writing to this address. R

    //8 + 8 + 16 + 8 + 64 = 104 bits per core. Max 9 - 1 cores. 104 * (9 - 1) / 8 = 104.
    uint8_t CORE_RESERVED[104];

    // ==========================================
    // Per-Channel MMIO Registers
    // Base Offset: 0x100 + (channel_id * 0x40)
    // ==========================================
    // Assumptions: 7 channel, max depth of 256 (min 8 bits).
    __O  uint8_t  START;                    // 0x00: Start channel operation. W
    __I  uint8_t  READY_STATUS;             // 0x02: Is channel ready. R
    __I  uint8_t  CHANNEL_BUFFER_COUNT;     // 0x03: # elements pending in channel (log₂(depth)). R
    __IO uint8_t  CORE_ID;                  // 0x04: mhartid of core loading this transaction. RW
    __IO uint16_t TRANSACTION_ID;           // 0x08: Desired transaction ID. RW
    __IO uint8_t  PERIPHERAL_ID;            // 0x0A: Peripheral ID to read/write. RW
    __IO uint8_t  TRANSACTION_PRIORITY;     // 0x0C: Transaction priority. RW
    __IO uint8_t  MODE;                     // 0x0E: Mode (bit 0 = addr gating, bit 1 = interrupts). RW
    __IO uint64_t ADDR_READ;                // 0x10: Read address. RW
    __IO uint64_t ADDR_WRITE;               // 0x18: Write address. RW
    __IO uint16_t NUM_PACKETS;              // 0x20: Number of packets to transfer. RW
    __IO uint8_t  LG_WIDTH;                 // 0x22: Log of # bytes per packet. RW
    __IO uint16_t READ_STRIDE;              // 0x24: Stride between read packets. RW
    __IO uint16_t WRITE_STRIDE;             // 0x26: Stride between write packets. RW
    __IO uint8_t  BUSY;                     // 0x28: If high, another core is writing to this channel. RW

    //264 bits per channel, max 7 channel. 264 * (7 - 1) / 8 = 198
    uint8_t CHANNEL_RESERVED[198];
  } DMA_Type;

  typedef struct {
    uint8_t core;
    uint16_t transaction_id;
    uint8_t transaction_priority;
    uint8_t peripheral_id;
    uint64_t addr_r;
    uint64_t addr_w;
    uint16_t inc_r;
    uint16_t inc_w;
    uint16_t len;
    uint8_t logw;
    bool do_interrupt;
    bool do_address_gate;
  } dma_transaction_t;

  //Default functions
  void reg_write8(uintptr_t addr, uint8_t data);
  uint8_t reg_read8(uintptr_t addr);
  void reg_write16(uintptr_t addr, uint16_t data);
  uint16_t reg_read16(uintptr_t addr);
  void reg_write32(uintptr_t addr, uint32_t data);
  uint32_t reg_read32(uintptr_t addr);
  void reg_write64(unsigned long addr, uint64_t data);
  uint64_t reg_read64(unsigned long addr);

  //DMA specific functions
  void setup_interrupts();

  bool set_DMA_C(uint32_t channel, dma_transaction_t transaction, bool retry);
  bool set_DMA_P(uint32_t channel, dma_transaction_t transaction, bool retry);
  
  void start_DMA(uint32_t channel, uint16_t transaction_id, bool* finished);
  
  #define ALWAYS_PRINT 0
  #define PRINT_ON_ERROR 1
  #define NEVER_PRINT 2
  
  bool check_val8(int i, unsigned int ref, long unsigned int addr, int print);
  bool check_val16(int i, unsigned int ref, long unsigned int addr, int print);
  bool check_val32(int i, unsigned int ref, long unsigned int addr, int print);
  bool check_val64(int i, long unsigned int ref, long unsigned int addr, int print);
  
  int dma_status();
  void dma_reset();
  
  void dma_wait_till_inactive(int cycle_no_inflight);
  void dma_wait_till_interrupt(bool* finished);
  void dma_wait_till_done(size_t mhartid, bool* finished);
  
  size_t ticks();
  
  void end_dma();

  #ifdef __cplusplus
  }
  #endif

  #endif // __HAL_DMA_H__