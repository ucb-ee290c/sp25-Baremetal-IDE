#ifndef MFCC_TEST_MAIN_H
#define MFCC_TEST_MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "riscv.h"
#include "riscv_math.h"

void app_init(void);
void app_main(void);

#ifdef __cplusplus
}
#endif

#endif
