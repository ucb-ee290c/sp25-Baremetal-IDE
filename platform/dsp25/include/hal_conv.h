 #ifndef __HAL_CONV_H__
 #define __HAL_CONV_H__
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 // Baremetal IDE Definitions //
 #include  "hal_mmio.h"
 #include "chip_config.h"
 
 #include <stdint.h>
 
 // Register Offset Definitions
 #define CONV_INPUT              0x00
 #define CONV_OUTPUT             0x20
 #define CONV_KERNEL             0x40
 #define CONV_STATUS             0x6A
 #define CONV_START              0x6C
 #define CONV_COUNT              0x70
 #define CONV_LENGTH             0x78
 #define CONV_DILATION           0x7C
 #define CONV_READ_CHECK         0x8D
 #define CONV_KERNEL_LEN         0x8E
 #define CONV_MMIO_RESET         0x8F
 
 void conv_init();
 
 int conv_set_params(uint32_t* input, uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length);
 
 void conv_read_output(uint32_t *output, int output_len, int status, uint32_t* input);
 
 void start_conv();
 
 uint8_t get_register_status();
 
 uint32_t get_register_out_count();
 
 uint8_t get_register_read_check();
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // __HAL_CONV_H__