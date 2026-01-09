#include "hal_wavelet.h"
#include "hal_mmio.h"

void wavelet_init() {
	reg_write32(WAVELET_EVEN_OUTPUT, 0);
	reg_write32(WAVELET_ODD_OUTPUT, 0);
	reg_write32(WAVELET_EVEN_INPUT, 0);
	reg_write32(WAVELET_ODD_INPUT, 0);
	reg_write32(WAVELET_START, 0);
	reg_write32(WAVELET_INPUT_FLAGS, 0);
	reg_write32(WAVELET_SEL, 0);
}

void start_wavelet() {
    reg_write8(WAVELET_START, 1);
}

uint64_t pack_samples(uint32_t odd_sample, uint32_t even_sample) {
    return ((uint64_t)odd_sample << 32) | even_sample;
}

void unpack_samples(uint64_t packed, uint32_t* odd_sample, uint32_t* even_sample) {
    *even_sample = (uint32_t)(packed & 0xFFFFFFFF);
    *odd_sample = (uint32_t)(packed >> 32);
}


void wavelet_forward(uint64_t *input_sample, uint8_t num_tests, uint64_t *output_sample, uint8_t sel) {
	reg_write32(WAVELET_INPUT_FLAGS, FORWARD);
    for (int i = 0; i < num_tests; i += 2) {
		reg_write32(WAVELET_EVEN_INPUT, input_sample[i]);
		reg_write32(WAVELET_ODD_INPUT, input_sample[i+1]);

		output_sample[i] = reg_read32(WAVELET_EVEN_OUTPUT);
		output_sample[i+1] = reg_read32(WAVELET_ODD_OUTPUT);
	}
}

void wavelet_inverse(uint64_t *input_sample, uint8_t num_tests, uint64_t *output_sample, uint8_t sel) {
	reg_write32(WAVELET_INPUT_FLAGS, INVERSE);
	for (int i = 0; i < num_tests; i += 2) {
		reg_write64(WAVELET_EVEN_INPUT, input_sample[i]);
		reg_write64(WAVELET_ODD_INPUT, input_sample[i+1]);
    
    	
		start_wavelet();

		output_sample[i] = reg_read64(WAVELET_EVEN_OUTPUT);
		output_sample[i+1] = reg_read64(WAVELET_ODD_OUTPUT);
	}
}