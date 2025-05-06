#include "hal_dma.h"
#include "chip_config.h"
#include <stdio.h>
#include <stdlib.h>

// Base registers - Use proper C syntax for constants with parentheses
#define DMA_MMIO_BASE (0x8812000UL)
#define DMA_RESET (DMA_MMIO_BASE)
#define DMA_INFLIGHT_STATUS (DMA_MMIO_BASE + 0x1UL)

// Helper constants - Add parentheses for all address calculations
#define CHANNEL_BASE (DMA_MMIO_BASE + 0x100UL)
#define CHANNEL_OFFSET (64UL)
#define INTERRUPT_BASE (DMA_MMIO_BASE + 16UL)
#define INTERRUPT_OFFSET (16UL)
#define DMA_INT_CORE_0_OFFSET (INTERRUPT_OFFSET * 0UL) // core 0
#define DMA_INT_CORE_1_OFFSET (INTERRUPT_OFFSET * 1UL) // core 1

// Per core registers - Add parentheses and UL suffix for all constants
#define DMA_INT_SERVICED (INTERRUPT_BASE + 0x00UL)
#define DMA_INT_VALID (INTERRUPT_BASE + 0x01UL)
#define DMA_INT_TRANSACTION_ID (INTERRUPT_BASE + 0x02UL)
#define DMA_INT_IS_ERROR (INTERRUPT_BASE + 0x04UL)
#define DMA_INT_ADDRESS (INTERRUPT_BASE + 0x08UL)

// Per channel registers - Add parentheses for all address calculations
#define DMA_START (CHANNEL_BASE)
#define DMA_BUSY (DMA_START + 0x28UL)
#define DMA_READY (DMA_START + 0x2UL)
#define DMA_FIFO_LENGTH (DMA_START + 0x3UL)
#define DMA_CORE_ID (DMA_START + 0x4UL)
#define DMA_TRANSACTION_ID (DMA_START + 0x8UL)
#define DMA_PERIPHERAL_ID (DMA_START + 0xAUL)
#define DMA_TRANSACTION_PRIORITY (DMA_START + 0xCUL)
#define DMA_MODE (DMA_START + 0xEUL)
#define DMA_ADDR_R (DMA_START + 0x10UL)
#define DMA_ADDR_W (DMA_START + 0x18UL)
#define DMA_LEN (DMA_START + 0x20UL)
#define DMA_LOGW (DMA_START + 0x22UL)
#define DMA_INC_R (DMA_START + 0x24UL)
#define DMA_INC_W (DMA_START + 0x26UL)

// interrupt-specific constants
#define DMA_INTERRUPT_ID_CORE_0 (4)
#define DMA_INTERRUPT_ID_CORE_1 (5)

// PLIC MMIO - Add UL suffix and parentheses
#define PLIC_BASE (0x0C000000UL)
#define PLIC_ENABLE_CORE_0 (PLIC_BASE + 0x2000UL)
#define PLIC_ENABLE_CORE_1 (PLIC_BASE + 0x2080UL)
#define PLIC_PRIORITY_DMA_INT_SRC_1 (PLIC_BASE + (4UL * DMA_INTERRUPT_ID_CORE_0))
#define PLIC_PRIORITY_DMA_INT_SRC_2 (PLIC_BASE + (4UL * DMA_INTERRUPT_ID_CORE_1))
#define PLIC_CLAIM_CORE_0 (PLIC_BASE + 0x200004UL) 
#define PLIC_CLAIM_CORE_1 (PLIC_BASE + 0x201004UL) 

// -------------- DMA specific functions ----------------------
typedef struct Node {
    uint16_t transaction_id;
    bool* complete;
    bool* error;
    struct Node* next;
} Node;

Node* create_node(uint16_t transaction_id, bool* complete) {
    Node* new_node = (Node*)malloc(sizeof(Node));
    if (!new_node) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    new_node->transaction_id = transaction_id;
    new_node->complete = complete;
    new_node->next = NULL;
    return new_node;
}
void add_tail(Node** head, uint16_t transaction_id, bool* complete) {
    Node* new_node = create_node(transaction_id, complete);
    if (*head == NULL) *head = new_node;
    else {
        Node* c = *head;
        while (c->next != NULL) c = c->next;
        c->next = new_node;
    }
}
bool set_complete(Node** head, uint16_t transaction_id) {
    Node* current = *head;
    Node* prev = NULL;
    while (current != NULL && current->transaction_id != transaction_id) {
        prev = current;
        current = current->next;
    }
    if (current == NULL) return false;
    *(current->complete) = true;

    if (prev == NULL) {
        *head = current->next;
    } else {
        prev->next = current->next;
    }
    free(current);
    return true;
}
bool check_complete(Node** head, uint16_t transaction_id) {
    Node* current = *head;
    while (current != NULL && current->transaction_id != transaction_id) {
        current = current->next;
    }
    if (current == NULL) return false;
    return *current->complete;
}
void free_list(Node* head) {
    Node* current = head;
    while (current != NULL) {
        Node* next = current->next;
        free(current);
        current = next;
    }
}
void print_status(Node* head) {
    Node* current = head;
    while (current != NULL) {
        printf("%d %d", *(current->complete), current->transaction_id);
        current = current->next;
    }
    printf("\n");
}

Node* complete_tracker_list = NULL;

void enable_interrupts() {
    // Enable interrupts on the PLIC for specific core
    reg_write32(PLIC_ENABLE_CORE_0, (1 << DMA_INTERRUPT_ID_CORE_0));
    reg_write32(PLIC_ENABLE_CORE_1, (1 << DMA_INTERRUPT_ID_CORE_1));

    // Set priority for DMA interrupts --  [1,7]) 1 - Low; 7 - High
    reg_write32(PLIC_PRIORITY_DMA_INT_SRC_1, 5);
    reg_write32(PLIC_PRIORITY_DMA_INT_SRC_2, 5);

    set_csr(mie, MIP_MEIP); // Enables external machine interrupts
    set_csr(mstatus, MSTATUS_MIE); // Enables machine interrupts
}

/*
    Handles the external machine interrupts.
    You can add more cases depending on the interrupt ID (external device) 
    read from the claim register in the PLIC.
*/

void machine_ext_interrupt_handler() {
    /*
        Code here will likely break if you are using more than 2 cores.
        This stuff could use refactoring to use for N cores, but DSP'25 uses
        2 cores so this works :)    
    */

    bool is_core_0 = read_csr(mhartid) == 0;
 
    // Read the interrupt source from PLIC
    uint32_t interrupt_id_core_0 = reg_read32(PLIC_CLAIM_CORE_0);
    uint32_t interrupt_id_core_1 = reg_read32(PLIC_CLAIM_CORE_1);

    if ((interrupt_id_core_0 == DMA_INTERRUPT_ID_CORE_0) ||
        (interrupt_id_core_1 == DMA_INTERRUPT_ID_CORE_1)) {  

        uint32_t CORE_OFFSET = is_core_0 ? DMA_INT_CORE_0_OFFSET : DMA_INT_CORE_1_OFFSET;
        uint32_t interrupt_id = is_core_0 ? interrupt_id_core_0 : interrupt_id_core_1;
        uint32_t PLIC_CLAIM_ADDRESS = is_core_0 ? PLIC_CLAIM_CORE_0 : PLIC_CLAIM_CORE_1;
        
        // Acknowledge interrupt
        uint16_t t_id = reg_read16(DMA_INT_TRANSACTION_ID + CORE_OFFSET);
        // TODO: some way to acknowledge this if it's an error?
        // uint64_t mem_address = reg_read64(DMA_INT_ADDRESS + CORE_OFFSET);
        // uint8_t is_error = reg_read8(DMA_INT_IS_ERROR + CORE_OFFSET);

        // Write back to PLIC to complete interrupt handling
        //  - If the completion ID does not match an interrupt source that is currently 
        //    enabled for the target, the completion is silently ignored
        
        reg_write32(PLIC_CLAIM_ADDRESS, interrupt_id);
        reg_write8(DMA_INT_SERVICED + CORE_OFFSET, 1); // Notify DMA that interrupt has been serviced

        set_complete(&complete_tracker_list, t_id);
    }
    else {
        printf("[WARNING] No external machine interrupts were serviced.\n");
    }
}


/*
    General trap handler function
    Can add different type of interrupt handlers here.
    
    * Need to make sure trap_handler is 4-byte aligned.
*/
uint64_t MCAUSE_INT      = 0x8000000000000000;   // if the 63rd bit is 1, then trap is caused by an interrupt
uint64_t MCAUSE_EXTERNAL = 0xB;                  // machine external interrupt exeception code is 11 which is set in mcause[9:0]
__attribute__((aligned(4))) void* trap_handler(uintptr_t epc, uintptr_t cause, uintptr_t tval, uintptr_t regs[32]) {
   
    if ((cause & MCAUSE_INT) && (cause & MCAUSE_EXTERNAL)) {
        machine_ext_interrupt_handler();
    } else {
        int code = cause & ((1UL << ((sizeof(int)<<3)-1)) - 1);
        code = ((intptr_t)cause < 0) ? -code : code;
        exit(code);
        // printf("[WARNING] Didn't service the trap.\n");
    }
    return (void*)epc;
    // asm volatile ("mret");
}

extern void trap_entry();

/*
    This function sets our trap handler as the jumping of any triggered trap.
*/
void setup_trap_handler() {
    uint64_t mtvec_value = (uint64_t)trap_entry;

    mtvec_value &= ~0x3; // Set mtvec fields 00 last two bits; says we are using a global trap handler
    write_csr(mtvec, mtvec_value);
}

void setup_interrupts() {
    enable_interrupts();
    setup_trap_handler();
}

bool set_DMA_common(uint32_t channel, dma_transaction_t transaction, bool retry) {
    uintptr_t channel_offset = channel * CHANNEL_OFFSET;

    // spin while another core is writing
    uint32_t res = reg_amo_swap32(DMA_BUSY + channel_offset, 1);
    if (res && !retry) return false;
    while (res && retry && reg_amo_swap32(DMA_BUSY + channel_offset, 1)) ;

    reg_write8(DMA_CORE_ID + channel_offset, transaction.core);
    reg_write16(DMA_TRANSACTION_ID + channel_offset, transaction.transaction_id);
    reg_write8(DMA_TRANSACTION_PRIORITY + channel_offset, transaction.transaction_priority);
    reg_write8(DMA_LOGW + channel_offset, transaction.logw);
    reg_write64(DMA_ADDR_R + channel_offset, transaction.addr_r);
    reg_write64(DMA_ADDR_W + channel_offset, transaction.addr_w);
    reg_write16(DMA_LEN + channel_offset, transaction.len);

    return true;
}

bool set_DMA_C(uint32_t channel, dma_transaction_t transaction, bool retry) {
    uintptr_t channel_offset = channel * CHANNEL_OFFSET;
    
    if (!set_DMA_common(channel, transaction, retry)) return false;

    reg_write16(DMA_INC_R + channel_offset, transaction.inc_r);
    reg_write16(DMA_INC_W + channel_offset, transaction.inc_w);

    // 1 by default
    uint8_t mode = transaction.do_interrupt + (transaction.do_address_gate << 1); 
    if (mode != 1) reg_write8(DMA_MODE + channel_offset, mode);

    return true;
}

bool set_DMA_P(uint32_t channel, dma_transaction_t transaction, bool retry) {
    uintptr_t channel_offset = channel * CHANNEL_OFFSET;

    if (!set_DMA_common(channel, transaction, retry)) return false;

    // these fields are 0 by default
    if (transaction.inc_r != 0) reg_write16(DMA_INC_R + channel_offset, transaction.inc_r);
    if (transaction.inc_w != 0) reg_write16(DMA_INC_W + channel_offset, transaction.inc_w);

    reg_write8(DMA_PERIPHERAL_ID + channel_offset, transaction.peripheral_id);

    // 3 by default
    uint8_t mode = transaction.do_interrupt + (transaction.do_address_gate << 1); 
    if (mode != 3) reg_write8(DMA_MODE + channel_offset, mode);

    return true;
}

void start_DMA(uint32_t channel, uint16_t transaction_id, bool* finished) {
    uintptr_t channel_offset = channel * CHANNEL_OFFSET;

    // Add to linked list
    if (finished) add_tail(&complete_tracker_list, transaction_id, finished);

    // Start DMA transaction
    reg_write8(DMA_START + channel_offset, 1);
}

void end_dma() {
    free_list(complete_tracker_list);
}

#define ALWAYS_PRINT 0
#define PRINT_ON_ERROR 1
#define NEVER_PRINT 2

#define CHECK_VAL_BODY(bits) \
{ \
    unsigned int poll = reg_read##bits(addr); \
    if (poll != ref) { \
        if (print == PRINT_ON_ERROR || print == ALWAYS_PRINT) \
            printf("[Transaction %d] %x does not match reference value %x at addr: [%lx]\n", i, poll, ref, addr); \
        return 1; \
    } else { \
        if (print == ALWAYS_PRINT) \
            printf("[Transaction %d] Success: (%x)\n", i, ref); \
    } \
    return 0; \
}

bool check_val8(int i, unsigned int ref, long unsigned int addr, int print) {
    CHECK_VAL_BODY(8)
}
bool check_val16(int i, unsigned int ref, long unsigned int addr, int print) {
    CHECK_VAL_BODY(16)
}
bool check_val32(int i, unsigned int ref, long unsigned int addr, int print) {
    CHECK_VAL_BODY(32)
}
bool check_val64(int i, long unsigned int ref, long unsigned int addr, int print) {
    long unsigned int poll = reg_read64(addr);
    if (poll != ref) {
        if (print == PRINT_ON_ERROR || print == ALWAYS_PRINT) printf("[%d]Hardware result (1) %lx does not match reference value %lx at addr: [%lx]\n", i, poll, ref, addr);
        return 1;
    } else {
        if (print == ALWAYS_PRINT) printf("[%d]Success: (%lx)\n", i, ref);
    }
    return 0;
}
size_t ticks() {
    return read_csr(mcycle);
} 
int dma_status() {
    return reg_read8(DMA_INFLIGHT_STATUS);
}
void dma_reset() {
    reg_write8(DMA_RESET, 1);
}
void dma_wait_till_inactive(int cycle_no_inflight) {
    while (1) {
        volatile int t = 0;
        while (t < cycle_no_inflight && dma_status() == 0) t++;
        if (t == cycle_no_inflight) break;
    }
}
void dma_wait_till_interrupt(bool* finished) {
    volatile int k = 0;
    while (!*finished) { k = k + 1; }
}

void dma_wait_till_done(size_t mhartid, bool* finished) {
    int core_offset = INTERRUPT_OFFSET * mhartid;
    while (!(*finished)) {
        volatile uint8_t int_done = reg_read8(DMA_INT_VALID + core_offset);
        if (int_done) {
            uint16_t t_id = reg_read16(DMA_INT_TRANSACTION_ID + core_offset);
            set_complete(&complete_tracker_list, t_id);
        }
    }
}
