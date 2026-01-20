#ifndef __SIMPLE_SETUP_H
#define __SIMPLE_SETUP_H

#include <stdint.h>

#include "chip_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_test(uint64_t target_frequency);
uint64_t rdcycle(void);

#ifdef __cplusplus
}
#endif

#endif /* __SIMPLE_SETUP_H */
