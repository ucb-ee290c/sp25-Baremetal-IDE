#ifndef TINYSPEECH_TEST_MAIN_H
#define TINYSPEECH_TEST_MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef TINYSPEECH_DEBUG_TRACE
#define TINYSPEECH_DEBUG_TRACE 1
#endif

void app_init(void);
void app_main(void);

#ifdef __cplusplus
}
#endif

#endif
