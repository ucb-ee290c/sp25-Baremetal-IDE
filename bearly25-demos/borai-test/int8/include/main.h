/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "hardware.h"

// 15M version
// #include "stories15M_ng.h"
// #include "tokenizer.h"

// 260K version
#include "stories260kq.h"
#include "tok512.h"

// NO ---------------- newer tok32000 stories260 version
// #include "weights_15Mq.h"
//#include "weights_260Kq_32000.h"
//#include "tokenizer_32000.h"


// Bora Datasets
// #include "bora_tok8096.h"
// #include "bora_260K8096.h"
//#include "bora_3M8096.h"
//#include "bora_15M8096.h"
//#include "bora_42M8096.h"


/**
 * This section controls which peripheral device is included in the application program.
 * To save the memory space, the unused peripheral device can be commented out.
 */
// #include "hal_core.h"
// #include "hal_clint.h"
// #include "hal_gpio.h"
// #include "hal_i2c.h"
// #include "hal_plic.h"
// #include "hal_uart.h"

/* USER CODE END Includes */

/* Private defines -----------------------------------------------------------*/
/* USER CODE BEGIN Private defines */
#define MODEL_MAGIC_NUMBER 0x616b3432
#define MODEL_VERSION_INT8 2
#define MODEL_V2_HEADER_SIZE 256

/* USER CODE END Private defines */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */
typedef enum {
    GENERATE,
    CHAT
} GenMode;
/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
/* USER CODE BEGIN EFP */

/*
 * TransformerWeightsT — transposed & packed int8 weight matrices.
 *
 * Each field is a B_pack array for the corresponding weight W[n_out × n_in]:
 *   B_pack layout: [(n_in+1) × n_out] bytes
 *     Row 0       : n_out zero bytes  (zero bias, no zero-point correction)
 *     Rows 1..K   : n_out int8 bytes per row (rows of W_T, pre-converted
 *                   from uint8 to int8 by subtracting 128 at startup)
 *
 * Enabled when -DTRANSPOSED_WEIGHTS is passed at compile time.
 * Allocated once in build_transformer(); freed in free_transformer().
 *
 * Memory overhead: total weight bytes × 1 (same element count as originals,
 * just reordered) + 1 extra bias row per weight matrix (negligible).
 */
#ifdef TRANSPOSED_WEIGHTS

typedef struct {
    unsigned char* wq_T;    /* (n_layers, dim+1, dim)         B_pack for wq  */
    unsigned char* wk_T;    /* (n_layers, dim+1, kv_dim)      B_pack for wk  */
    unsigned char* wv_T;    /* (n_layers, dim+1, kv_dim)      B_pack for wv  */
    unsigned char* wo_T;    /* (n_layers, dim+1, dim)         B_pack for wo  */
    unsigned char* w1_T;    /* (n_layers, dim+1, hidden_dim)  B_pack for w1  */
    unsigned char* w20_T;   /* (n_layers, hidden_dim+1, dim0) B_pack for w2 first half */
    unsigned char* w21_T;   /* (n_layers, hidden_dim+1, dim1) B_pack for w2 second half */
    int w20_n;              /* dim0 = floor(dim/2) */
    int w21_n;              /* dim1 = dim - dim0 */
    unsigned char* w3_T;    /* (n_layers, dim+1, hidden_dim)  B_pack for w3  */
    unsigned char* wcls0_T; /* (dim+1, vocab0)                B_pack for wcls first half */
    unsigned char* wcls1_T; /* (dim+1, vocab1)                B_pack for wcls second half */
    int wcls0_n;            /* vocab0 = floor(vocab_size/2) */
    int wcls1_n;            /* vocab1 = vocab_size - vocab0 */
} TransformerWeightsT;
#endif /* TRANSPOSED_WEIGHTS */

int main(int argc, char** argv);
void __main();
/* USER CODE END EFP */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
