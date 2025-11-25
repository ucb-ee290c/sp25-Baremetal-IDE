/* This code should run a hello world which will be the simplest test when 
   working with a bitstream and uploading a binary or when working on a chip 
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
