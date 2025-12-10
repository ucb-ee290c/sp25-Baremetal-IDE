#ifndef HAL_CAN_H
#define HAL_CAN_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chip_config.h"

// ================================
//  MMIO Address Definitions
// ================================

// MMIO Offsets
#define CAN_OFFSET_TX_EN        0x00 // TX Enable (1-bit, R/W)
#define CAN_OFFSET_RX_EN        0x04 // RX Enable (1-bit, W)
#define CAN_OFFSET_CLK_CONFIG   0x08 // BRP, TS1, TS2 (R/W)

#define CAN_OFFSET_TX_HDR       0x10 // TX Message Header (std_id, ext_id_lsb, ide, rtr, txrq) (R/W)
#define CAN_OFFSET_TX_DLC       0x14 // TX Data Length Code (DLC) (4-bit, R/W)
#define CAN_OFFSET_TX_DATA      0x18 // TX Data Payload (64-bit, R/W)
#define CAN_OFFSET_TX_OK        0x20 // TX Transmission Finished Flag (1-bit, R)

#define CAN_OFFSET_RX_HDR       0x40 // RX Message Header (std_id, ext_id_lsb, ide, rtr, rxmp) (R)
#define CAN_OFFSET_RX_DLC       0x44 // RX Data Length Code (DLC) (4-bit, R)
#define CAN_OFFSET_RX_DATA      0x48 // RX Data Payload (64-bit, R)
#define CAN_OFFSET_RX_ACK       0x50 // RX Message Acknowledge (1-bit, W)

#define CAN_OFFSET_STATE        0x80 // Debug: TX and RX State Machines (R)

// Bit Widths for Registers
#define CAN_WIDTH_STD_ID 11
#define CAN_WIDTH_EXT_ID 18
#define CAN_WIDTH_IDE 1
#define CAN_WIDTH_RTR 1
#define CAN_WIDTH_DLC 4

#define CAN_WIDTH_BRP 10
#define CAN_WIDTH_TS1 4
#define CAN_WIDTH_TS2 3

// Bit Masks and Shifts for State and RX Header Registers
#define CAN_TX_STATE_MASK 0b11111 // 5 bits for TX state
#define CAN_RX_STATE_MASK 0b1111100000 // 5 bits for RX state, shifted by 5

#define CAN_RX_STD_ID_MASK 0b11111111111
#define CAN_RX_EXT_ID_MASK 0b111111111111111111
#define CAN_RX_IDE_MASK 0b1
#define CAN_RX_RTR_MASK 0b1
#define CAN_RX_RXMP_MASK 0b1 // Message pending flag is bit 31 in 0x40, but is a standalone field in the Scala

// TX Request Bit (tx_trigger is RegField at 0x10, bit 32)
#define CAN_TX_HDR_TXRQ_BIT 31
#define CAN_TX_HDR_TXRQ_MASK (1U << CAN_TX_HDR_TXRQ_BIT)


// ================================
//  Data Structures
// ================================

// A union for flexible access to the 8-byte data payload
typedef union {
    uint8_t arr[8];
    uint64_t value; // The 64-bit value written to/read from the MMIO register
} CANData;

// The structure for a CAN Message
typedef struct {
    // Standard ID (11 bits) or Extended ID MSBs (11 bits)
    uint16_t std_id;         // (0x10, bits 0-10) / (0x40, bits 0-10)
    uint32_t ext_id_lsb;     // Extended ID LSBs (18 bits) (0x10, bits 11-28) / (0x40, bits 11-28)
    uint8_t ide;             // ID Extension Flag (1-bit) (0x10, bit 29) / (0x40, bit 29)
    uint8_t rtr;             // Remote Transmission Request (1-bit) (0x10, bit 30) / (0x40, bit 30)
    uint8_t dlc;             // Data Length Code (4-bit, max 8) (0x14/0x44, bits 0-3)
    CANData *data;           // Pointer to the 8-byte data payload
} CANMessage;

// The structure for configuring the bit timing
typedef struct {
    uint16_t brp; // Bit rate prescaler (10-bit)
    uint8_t ts1;  // Time segment 1 duration (4-bit)
    uint8_t ts2;  // Time segment 2 duration (3-bit)
} CANClockConfig;

// Enumeration/String for CAN State Machine Debugging
typedef char* CANState;

static const CANState CAN_STATES[] = {
    "NONE", "START_OF_FRAME", "ID_A", "SRR", "IDE", "ID_B", "RTR", "R0", "R1",
    "DLC", "DATA", "CRC", "CRC_DELIMITER", "ACK_SLOT", "ACK_DELIMITER",
    "END_OF_FILE", "INTER_FRAME_SPACING", "ERROR"
};

// ===================================
//  CAN Driver Function Prototypes
// ===================================

/**
 * @brief Sets the bit timing configuration (BRP, TS1, TS2).
 *
 * @param config Pointer to the CANClockConfig structure.
 */
void can_set_clock_config(CANClockConfig *config);

/**
 * @brief Enables or disables the CAN Transmit (TX) path.
 *
 * @param en Enable (true) or disable (false) the TX path.
 */
void can_tx_enable(bool en);

/**
 * @brief Enables or disables the CAN Receive (RX) path.
 *
 * @param en Enable (true) or disable (false) the RX path.
 */
void can_rx_enable(bool en);

/**
 * @brief Writes a CAN message to the peripheral for transmission.
 *
 * This function handles writing the header, DLC, and data, and triggers
 * the transmission request (TXRQ).
 *
 * @param msg Pointer to the CANMessage structure to transmit.
 * @param blocking If true, waits until the previous message is sent before writing.
 */
void can_write_msg(CANMessage *msg, bool blocking);

/**
 * @brief Reads a received CAN message from the peripheral.
 *
 * This function reads the header, DLC, and data, and then acknowledges (ACKs)
 * the message to clear the 'message pending' flag.
 *
 * @param msg Pointer to the CANMessage structure to store the received data.
 * @param blocking If true, waits until a message is pending before reading.
 */
void can_read_msg(CANMessage *msg, bool blocking);

/**
 * @brief Checks if the last transmitted message has completed.
 *
 * @return true if transmission has finished (tx_txok is high), false otherwise.
 */
bool can_tx_msg_sent(void);

/**
 * @brief Checks if a new message is pending in the RX buffer.
 *
 * @return true if a message is pending (rx_rxmp is high), false otherwise.
 */
bool can_rx_msg_pending(void);

/**
 * @brief Acknowledges that the pending message has been read.
 *
 * This clears the 'message pending' flag (rx_rxmp) and allows the next
 * received message to be loaded into the RX registers.
 */
void can_rx_msg_ack(void);

/**
 * @brief Gets the current state of the CAN TX state machine (debug).
 *
 * @return A constant string describing the current TX state.
 */
const CANState can_tx_get_state(void);

/**
 * @brief Gets the current state of the CAN RX state machine (debug).
 *
 * @return A constant string describing the current RX state.
 */
const CANState can_rx_get_state(void);

#endif /* HAL_CAN_H */