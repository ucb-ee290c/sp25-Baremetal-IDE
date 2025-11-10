/* =========================================================================
 * main.h — Simple tests for OPE matmul
 * ========================================================================= */
#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hal_ope.h"
#include "hal_ope_prepack.h"

#ifndef PRINT_INPUT_MATRICES
#define PRINT_INPUT_MATRICES 0  // Set to 1 to print input A and B matrices
#endif

#ifndef PRINT_SUCCESS_MATRICES
#define PRINT_SUCCESS_MATRICES 0  // Set to 1 to print output matrices
#endif

#define OPE_PACK_WS_MAX_BYTES   (512 * 1024)

void app_init(void);
void app_main(void);

#endif /* __MAIN_H__ */
