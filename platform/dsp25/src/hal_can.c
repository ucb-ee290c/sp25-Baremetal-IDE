#include "hal_can.h"
#include <assert.h>

// Helper to convert offset to an absolute MMIO address
static uintptr_t offset_to_ptr(uint32_t offset) {
    return (uintptr_t) (CAN_BASE + offset);
}

void can_set_clock_config(CANClockConfig *config) {
    // The CLK_CONFIG register (0x08) packs BRP, TS1, and TS2.
    // BRP (10-bit) | TS1 (4-bit, shifted by 10) | TS2 (3-bit, shifted by 14)
    uint32_t cfg = config->brp |
                   (config->ts1 << CAN_WIDTH_BRP) |
                   (config->ts2 << (CAN_WIDTH_BRP + CAN_WIDTH_TS1));
    reg_write32(offset_to_ptr(CAN_OFFSET_CLK_CONFIG), cfg);
}

void can_tx_enable(bool en) {
    // TX_EN is a 1-bit register at 0x00
    reg_write8(offset_to_ptr(CAN_OFFSET_TX_EN), en ? 1 : 0);
}

void can_rx_enable(bool en) {
    // RX_EN is a 1-bit register at 0x04
    reg_write8(offset_to_ptr(CAN_OFFSET_RX_EN), en ? 1 : 0);
}

void can_write_msg(CANMessage *msg, bool blocking) {
    /* Ensure data is valid */
    assert(msg->dlc <= 8);
    assert(msg->data != NULL);

    /* Wait for last message transmit to finish */
    if (blocking) {
        // tx_txok is 1 when finished, 0 when active
        while (!can_tx_msg_sent());
    }
    // Assertion: if not blocking, ensure we are not currently busy
    assert(can_tx_msg_sent());

    /* Create message header to write to register 0x10 */
    // Note: The tx_trigger/TXRQ bit must be set in this write to start transmission.
    uint32_t hdr = (msg->std_id) |
        (msg->ext_id_lsb << CAN_WIDTH_STD_ID) |
        (msg->ide << (CAN_WIDTH_STD_ID + CAN_WIDTH_EXT_ID)) |
        (msg->rtr << (CAN_WIDTH_STD_ID + CAN_WIDTH_EXT_ID + CAN_WIDTH_IDE)) |
        (CAN_TX_HDR_TXRQ_MASK); // Set the TX Request bit (tx_trigger/TXRQ)

    /* Write data length to register 0x14 */
    reg_write32(offset_to_ptr(CAN_OFFSET_TX_DLC), msg->dlc);

    /* Write data to register 0x18 */
    reg_write64(offset_to_ptr(CAN_OFFSET_TX_DATA), msg->data->value);

    /* Write message header (including TXRQ) to register 0x10.
       This must occur *after* data store to trigger transmission of the new data. */
    reg_write32(offset_to_ptr(CAN_OFFSET_TX_HDR), hdr);

    /* Wait for message transmit to finish if blocking */
    if (blocking) {
        while (!can_tx_msg_sent());
    }
}

void can_read_msg(CANMessage *msg, bool blocking) {
    assert(msg->data != NULL);

    /* Wait for message to be pending (rx_rxmp high) */
    if (blocking) {
        while (!can_rx_msg_pending());
    }
    assert(can_rx_msg_pending());

    // Read all registers (order doesn't strictly matter for reads, but good practice)
    uint32_t hdr_raw = reg_read32(offset_to_ptr(CAN_OFFSET_RX_HDR));
    uint8_t dlc = reg_read8(offset_to_ptr(CAN_OFFSET_RX_DLC));
    uint64_t data = reg_read64(offset_to_ptr(CAN_OFFSET_RX_DATA));

    // Decode Header (0x40)
    msg->std_id = hdr_raw & CAN_RX_STD_ID_MASK;
    msg->ext_id_lsb = (hdr_raw >> CAN_WIDTH_STD_ID) & CAN_RX_EXT_ID_MASK;
    msg->ide = (hdr_raw >> (CAN_WIDTH_STD_ID + CAN_WIDTH_EXT_ID)) & CAN_RX_IDE_MASK;
    msg->rtr = (hdr_raw >> (CAN_WIDTH_STD_ID + CAN_WIDTH_EXT_ID + CAN_WIDTH_IDE)) & CAN_RX_RTR_MASK;
    // The rxmp bit is the next bit after rtr, but we check it separately in can_rx_msg_pending

    // Decode DLC (0x44) and Data (0x48)
    msg->dlc = dlc;
    msg->data->value = data;

    // Acknowledge the message read to allow the next message to be loaded (0x50)
    can_rx_msg_ack();
}

bool can_tx_msg_sent(void) {
    // TX_OK (0x20) is 1 when transmission is finished, 0 when active
    return reg_read8(offset_to_ptr(CAN_OFFSET_TX_OK)) & 0b1;
}

bool can_rx_msg_pending(void) {
    // RX_HDR (0x40) has the rxmp (Message Pending) flag as the last bit (bit 31 in the 32-bit reg)
    // The Scala definition says: RegField.r(1, impl.io.rxio.rxmp, ...)
    // Given the structure, rxmp should be the bit *after* rtr, which is bit 31 (if the register is 32 bits).
    // The definition from your existing can.c assumes it's bit 31, which is a common pattern:
    // `(reg_read32(offset_to_ptr(CAN_OFFSET_RX_HDR)) >> 31) & 0b1`

    // Bits in 0x40: 11 (std) + 18 (ext) + 1 (ide) + 1 (rtr) = 31 bits.
    // The 32nd bit (index 31) is the rxmp bit.
    const uint32_t RXMP_BIT_SHIFT = CAN_WIDTH_STD_ID + CAN_WIDTH_EXT_ID + CAN_WIDTH_IDE + CAN_WIDTH_RTR; // = 31
    return (reg_read32(offset_to_ptr(CAN_OFFSET_RX_HDR)) >> RXMP_BIT_SHIFT) & 0b1;
}

void can_rx_msg_ack(void) {
    // RX_ACK (0x50) is a write-only register to acknowledge a read
    // The Scala: RegField.w(1, rxack, ...). Writing 1 acknowledges.
    reg_write8(offset_to_ptr(CAN_OFFSET_RX_ACK), 0b1);
}

const CANState can_tx_get_state(void) {
    // STATE (0x80) has TX state (5 bits) in the LSBs
    uint32_t raw_state = reg_read32(offset_to_ptr(CAN_OFFSET_STATE));
    return CAN_STATES[raw_state & CAN_TX_STATE_MASK];
}

const CANState can_rx_get_state(void) {
    // STATE (0x80) has RX state (5 bits) in the next 5 bits (shifted by 5)
    // The Scala: RegField.r(5, impl.io.rxio.state, ...) which is after the 5 bits for tx_state.
    const uint32_t RX_STATE_SHIFT = 5;
    uint32_t raw_state = reg_read32(offset_to_ptr(CAN_OFFSET_STATE));
    return CAN_STATES[(raw_state >> RX_STATE_SHIFT) & CAN_TX_STATE_MASK]; // Use the same 5-bit mask
}