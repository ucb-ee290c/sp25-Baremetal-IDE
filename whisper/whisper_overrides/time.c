#include <stdint.h>
#include "chip_config.h"

// These functions all replace Whisper's time queries with the Baremetal equivalents
// instead of syscalls.

void ggml_time_init(void) {}

int64_t ggml_time_ms(void) {
    return CLINT->MTIME / MTIME_FREQ;
}

int64_t ggml_time_us(void) {
    return (CLINT->MTIME * 1000) / MTIME_FREQ;
}

int64_t ggml_cycles(void) {
    return CLINT->MTIME; // This should be equivalent to C's clock()... is this right?
}

int64_t ggml_cycles_per_ms(void) {
    return MTIME_FREQ / 1000;
}

 // I have no idea what this does, but it fixes some linker errors
 // This isn't time related, I just dumped it in here
void *__dso_handle = NULL;