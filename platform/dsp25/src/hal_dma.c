#include "hal_dma.h"
#include "chip_config.h"

//From mmio.h, removed static keyword since tests may use these functions
inline void reg_write8(uintptr_t addr, uint8_t data) {
    volatile uint8_t *ptr = (volatile uint8_t *)addr;
    *ptr = data;
}

inline uint8_t reg_read8(uintptr_t addr) {
    volatile uint8_t *ptr = (volatile uint8_t *)addr;
    return *ptr;
}
  
inline void reg_write16(uintptr_t addr, uint16_t data) {
    volatile uint16_t *ptr = (volatile uint16_t *)addr;
    *ptr = data;
}
  
inline uint16_t reg_read16(uintptr_t addr) {
    volatile uint16_t *ptr = (volatile uint16_t *)addr;
    return *ptr;
}
  
inline void reg_write32(uintptr_t addr, uint32_t data) {
    volatile uint32_t *ptr = (volatile uint32_t *)addr;
    *ptr = data;
}
  
inline uint32_t reg_read32(uintptr_t addr) {
    volatile uint32_t *ptr = (volatile uint32_t *)addr;
    return *ptr;
}
  
inline void reg_write64(unsigned long addr, uint64_t data) {
    volatile uint64_t *ptr = (volatile uint64_t *)addr;
    *ptr = data;
}
  
inline uint64_t reg_read64(unsigned long addr) {
    volatile uint64_t *ptr = (volatile uint64_t *)addr;
    return *ptr;
}

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
    reg_write32(PLIC_ENABLE_REG(0), (1 << DMA_INT_ID_CORE_0));
    reg_write32(PLIC_ENABLE_REG(1), (1 << DMA_INT_ID_CORE_1));

    // Set priority for DMA interrupts --  [1,7]) 1 - Low; 7 - High
    reg_write32(PLIC_PRIORITY_REG(DMA_INT_ID_CORE_0), 5);
    reg_write32(PLIC_PRIORITY_REG(DMA_INT_ID_CORE_1), 5);

    set_csr(mie, MIP_MEIP); // Enables external machine interrupts
    set_csr(mstatus, MSTATUS_MIE); // Enables machine interrupts
}

/*
    Handles the external machine interrupts.
    You can add more cases depending on the interrupt ID (external device) 
    read from the claim register in the PLIC.
*/
void machine_ext_interrupt_handler() {

    uint32_t core_id = read_csr(mhartid);
 
    // Read the interrupt source from PLIC
    uint32_t interrupt_id = reg_read32(PLIC_CLAIM_REG(core_id));

    if ((interrupt_id == DMA_INT_ID_CORE_0) ||
        (interrupt_id == DMA_INT_ID_CORE_1)) {  
        
        // Acknowledge interrupt
        uint16_t t_id = reg_read16(DMA_INT_TRANSACTION_ID(core_id));
        // TODO: some way to acknowledge this if it's an error?
        // uint64_t mem_address = reg_read64(DMA_INT_ADDRESS(core_id));
        // uint8_t is_error = reg_read8(DMA_INT_IS_ERROR(core_id));

        // Write back to PLIC to complete interrupt handling
        //  - If the completion ID does not match an interrupt source that is currently 
        //    enabled for the target, the completion is silently ignored
        
        reg_write32(PLIC_CLAIM_REG(core_id), interrupt_id);
        reg_write8(DMA_INT_SERVICED(core_id), 1); // Notify DMA that interrupt has been serviced

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
    return epc;
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
    if (!res && !retry) return false;
    while (!res && retry && reg_amo_swap32(DMA_BUSY + channel_offset, 1)) ;

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

    // Start DMA transaction
    reg_write8(DMA_START + channel_offset, 1);

    // Add to linked list
    if (finished) add_tail(&complete_tracker_list, transaction_id, finished);
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
        int t = 0;
        while (t < cycle_no_inflight && dma_status() == 0) t++;
        if (t == cycle_no_inflight) break;
    }
}
void dma_wait_till_interrupt(bool* finished) {
    volatile int k = 0;
    while (!*finished) { k = k + 1; }
}
