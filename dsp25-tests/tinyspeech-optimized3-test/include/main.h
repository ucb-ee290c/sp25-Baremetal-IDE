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
#define TINYSPEECH_DEBUG_TRACE 0
#endif

#ifndef TINYSPEECH_VERBOSE_CASE_LOGS
#define TINYSPEECH_VERBOSE_CASE_LOGS 0
#endif

#ifndef TINYSPEECH_PRINT_CASE_OUTPUTS
#define TINYSPEECH_PRINT_CASE_OUTPUTS 1
#endif

#ifndef TINYSPEECH_PRINT_CASE_PROBS
#define TINYSPEECH_PRINT_CASE_PROBS 1
#endif

#ifndef TINYSPEECH_EXPECTED_NUM_CASES
#define TINYSPEECH_EXPECTED_NUM_CASES 100
#endif

#ifndef TINYSPEECH_OUTPUT_SOFTMAX
#define TINYSPEECH_OUTPUT_SOFTMAX 0
#endif

#ifndef TINYSPEECH_ENABLE_TRACE
#define TINYSPEECH_ENABLE_TRACE 0
#endif

#ifndef TINYSPEECH_FP16_MIXED
#define TINYSPEECH_FP16_MIXED 0
#endif

void app_init(void);
int app_main(void);

#ifdef __cplusplus
}
#endif

#endif
