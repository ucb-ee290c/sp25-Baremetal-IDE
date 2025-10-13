/* =========================================================================
 * main.h — Simple tests for OPE matmul
 * ========================================================================= */
#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "hal_ope.h"

#ifdef __cplusplus
extern "C" {
#endif

void app_init(void);
void app_main(void);

#ifdef __cplusplus
}
#endif
#endif /* __MAIN_H__ */
