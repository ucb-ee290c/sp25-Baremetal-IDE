#include "hal_conv.h"
#include "chip_config.h"

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

void conv_init(ConvAccel_Type *conv) {
    reg_write64((uintptr_t)&conv->INPUT, 0);  
    reg_write64((uintptr_t)&conv->OUTPUT, 0);        
    reg_write64((uintptr_t)&conv->KERNEL, 0);        

    reg_write8((uintptr_t)&conv->START, 0);          
    reg_write32((uintptr_t)&conv->OUT_COUNT, 0);     
    reg_write32((uintptr_t)&conv->LENGTH, 0);        
    reg_write16((uintptr_t)&conv->DILATION, 0);      

    reg_write8((uintptr_t)&conv->KERNEL_LEN, 0);     
    reg_write8((uintptr_t)&conv->MMIO_RESET, 1);
}

int conv_set_params(ConvAccel_Type *conv, uint32_t* input, uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length){
    reg_write8((uintptr_t)&conv->MMIO_RESET, 0);

    for (int i = 0; i < input_length; i += 2) {
        reg_write64((uintptr_t)&conv->INPUT, *((uint64_t*) (input + i)));
    }

    reg_write32((uintptr_t)&conv->LENGTH, input_length);
    reg_write16((uintptr_t)&conv->DILATION, dilation);

    if (kernel_length == 8) {
        for (int i = 0; i < 8; i += 2) {
            reg_write64((uintptr_t)&conv->KERNEL, *((uint64_t*) (kernel + i)));
        }
        reg_write8((uintptr_t)&conv->KERNEL_LEN, 0);
    } else if (kernel_length == 16) {
        for (int i = 0; i < 16; i += 2) {
            reg_write64((uintptr_t)&conv->KERNEL, *((uint64_t*) (kernel + i)));
        }
        reg_write8((uintptr_t)&conv->KERNEL_LEN, 1);  
    } else {
        return -1;  
    }

    return 0;

}

void conv_read_output(ConvAccel_Type *conv, uint32_t *output, int output_len, int status, uint32_t* input) {
    int i = 0;

    // Read pairs of FP32s (2 per 64-bit read)
    for (; i < output_len/2 ; i += 2) {
        uint64_t current_out = reg_read64((uintptr_t)&conv->OUTPUT);
        uint32_t *unpacked = (uint32_t *) &current_out;

        if (i < 4) {
            reg_write64((uintptr_t)&conv->INPUT, *((uint64_t*) (input + (6 + 2*(i+1)))));
        }

        output[i]     = unpacked[0];
        output[i + 1] = unpacked[1];
    }

    // Final 1 read: 1 output
    uint64_t last_out = reg_read64((uintptr_t)&conv->OUTPUT);
    uint32_t* unpacked_out = (uint32_t*) &last_out;
    output[output_len - 1] = unpacked_out[0];
    
    status = reg_read8((uintptr_t)&conv->STATUS);
}

void start_conv(ConvAccel_Type *conv) {
    reg_write8((uintptr_t)&conv->START, 1);
}

uint8_t get_register_status(ConvAccel_Type *conv) {
    return reg_read8((uintptr_t)&conv->STATUS);
}

uint32_t get_register_out_count(ConvAccel_Type *conv) {
    return reg_read32((uintptr_t)&conv->OUT_COUNT);
}

uint8_t get_register_read_check(ConvAccel_Type *conv) {
    return reg_read8((uintptr_t)&conv->READ_CHECK);
}
