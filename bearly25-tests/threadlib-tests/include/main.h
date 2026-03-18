/*
 * main.h - Common includes and entry points for thread-lib tests.
 */
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "riscv.h"

void app_init();
void app_main();

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
