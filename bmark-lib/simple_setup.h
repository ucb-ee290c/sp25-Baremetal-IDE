#ifndef __SIMPLE_SETUP_H
#define __SIMPLE_SETUP_H

#include <stdint.h>
#include <unistd.h>

#include "chip_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_test(uint64_t target_frequency);
void reconfigure_pll(uint64_t target_frequency, uint32_t sleep_ms);
uint64_t rdcycle(void);
int msleep(useconds_t msec);

#ifdef __cplusplus
}
#endif

#endif /* __SIMPLE_SETUP_H */
