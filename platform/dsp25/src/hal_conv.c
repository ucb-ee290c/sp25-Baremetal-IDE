#include "hal_conv.h"
#include "hal_mmio.h"

void conv_init() {
    reg_write64(CONV_INPUT, 0);  
    reg_write64(CONV_OUTPUT, 0);        
    reg_write64(CONV_KERNEL, 0);        

    reg_write8(CONV_START, 0);          
    reg_write32(CONV_COUNT, 0);     
    reg_write32(CONV_LENGTH, 0);        
    reg_write16(CONV_DILATION, 0);      

    reg_write8(CONV_KERNEL_LEN, 0);     
    reg_write8(CONV_MMIO_RESET, 1);
}

int conv_set_params(uint32_t* input, uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length){
    reg_write8(CONV_MMIO_RESET, 0);
    reg_write8(CONV_START, 0);

    for (int i = 0; i < input_length; i += 2) {
        reg_write64(CONV_INPUT, *((uint64_t*) (input + i)));
    }

    reg_write32(CONV_LENGTH, input_length);
    reg_write16(CONV_DILATION, dilation);

    if (kernel_length == 8) {
        for (int i = 0; i < 8; i += 2) {
            reg_write64(CONV_KERNEL, *((uint64_t*) (kernel + i)));
        }
        reg_write8(CONV_KERNEL_LEN, 0);
    } else if (kernel_length == 16) {
        for (int i = 0; i < 16; i += 2) {
            reg_write64(CONV_KERNEL, *((uint64_t*) (kernel + i)));
        }
        reg_write8(CONV_KERNEL_LEN, 1);
    } else {
        return -1;  
    }

    return 0;

}

void conv_read_output(uint32_t *output, int output_len, int status, uint32_t* input) {
    int i = 0;

    // Read pairs of FP32s (2 per 64-bit read)
    for (; i < output_len/2 ; i += 2) {
        uint64_t current_out = reg_read64(CONV_OUTPUT);
        uint32_t *unpacked = (uint32_t *) &current_out;

        if (i < 4) {
            reg_write64(CONV_INPUT, *((uint64_t*) (input + (6 + 2*(i+1)))));
        }

        output[i]     = unpacked[0];
        output[i + 1] = unpacked[1];
    }

    // Final 1 read: 1 output
    uint64_t last_out = reg_read64(CONV_OUTPUT);
    uint32_t* unpacked_out = (uint32_t*) &last_out;
    output[output_len - 1] = unpacked_out[0];
    
    status = reg_read8(CONV_STATUS);
}

void start_conv() {
    reg_write8(CONV_START, 1);
}

uint8_t get_register_status() {
    return reg_read8(CONV_STATUS);
}

uint32_t get_register_out_count() {
    return reg_read32(CONV_COUNT);
}

uint8_t get_register_read_check() {
    return reg_read8(CONV_READ_CHECK);
}