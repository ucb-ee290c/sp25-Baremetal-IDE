// #include "hal_conv.h"
// #include "chip_config.h"

// // --- 1D Convolution Driver Functions ---

// void conv_init() {
//   reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  
//   reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 1);   
//   reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 1);
// }

// // Prefill the vector queues input data for a max 16 entries where each each is 2 x 32FP
// // Prefill the Kernel queue for either 
// // 
// // dilation, and kernel data. 
// int conv_set_params(uint32_t* input, uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length) {
//   // Clear MMIO Reset, Clear Datapath, Stop
//   reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 0);
//   reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 0); 
//   reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  

//   // Write input data (Streaming 64-bit writes)
//   for (int i = 0; i < input_length; i += 2) {
//     reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR), *((uint64_t*) (input + i)));
//   }

//   // Write parameters
//   reg_write32((uintptr_t)(MMIO_BASE + CONV_LENGTH_ADDR), input_length);
//   reg_write16((uintptr_t)(MMIO_BASE + CONV_DILATION_ADDR), dilation);

//   // Write kernel data and kernel length encoding
//   if (kernel_length == 8) {
//     for (int i = 0; i < 8; i += 2) {
//       reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
//     }
//     reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 0);
//   } else if (kernel_length == 16) {
//     for (int i = 0; i < 16; i += 2) {
//     reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
//     }
//     reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 1);   } 
//   else {
//     return -1;  
//   }

//   return 0;

// }

// uint8_t conv_read_output(uint32_t *output, uint32_t output_len) {

//   // Read pairs of FP32s (2 per 64-bit read)

//   int j; // Index for 32-bit output elements
//   for (j = 0; j < output_len - 1; j += 2) {
//     // TODO: implement checks to make sure there are things to read from the output queue
//     // if (get_register_out_count() == 0) {
//     //  return -1;
//     // }
//     uint64_t current_out = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
//     uint32_t *unpacked = (uint32_t *) &current_out;

//     output[j]   = unpacked[0];
//     output[j + 1] = unpacked[1];
//   }
    
//   // Handle the final odd element if output_len is odd
//   if (output_len % 2 != 0) {
//      uint64_t last_out = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
//      uint32_t* unpacked_out = (uint32_t*) &last_out;
//      output[output_len - 1] = unpacked_out[0];
//   }

//   return get_register_status();
// }

// void start_conv() {
//   reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 1);
// }

// uint8_t get_register_status() {
//   return reg_read8((uintptr_t)(MMIO_BASE + CONV_STATUS_ADDR));
// }

// uint32_t get_register_out_count() {
//   return reg_read32((uintptr_t)(MMIO_BASE + CONV_COUNT_ADDR));
// }

// uint8_t get_register_read_check() {
//   return reg_read8((uintptr_t)(MMIO_BASE + READ_CHECK_ADDR));
// }

// uint8_t perform_convolution_1D(uint32_t* input, uint32_t input_length, uint32_t* kernel, uint8_t kernel_length, uint32_t* output, uint16_t dilation) 
//   {

//   uint32_t output_len = input_length + kernel_length - 1;

//   // Checks:

//   // TODO: Check that the kernel length is less than or equal to ... How small is too small?

//   // TODO: Implement bigger than 16 kernel size! By implementing the following

//   // Note: about using this accelerator is that a key bottleneck may be the length of the kernel. In that case, it is advised to think about how shorter outputs can be stitched together.
//   // For example, convolutions (cross-correlations) can be done in blocks of 16, as that is the max kernel length, and stitched together

//   // for (size_t n = 0; n < temp_len; ++n) {
//   //           if (block_start + n < out_len)
//   //               output[block_start + n] += temp[n];}



//   // Initialize 1D Convolution Engine 
//   conv_init();
  
//   // TODO: Implement the following comment
//   // The below is for kernel lengths of 8 or 16 and input arrays of arbitrarily long length (minimum length being _____)
//   // for () {}

//   // Perform a single convolution for inputs an arbitrary number of FP32 values and kernels of either 8 or 16 FP32 values.



//   // Performs a single convolution for inputs up to 32 FP32 values and kernels of either 8 or 16 FP32 values.

//   // Prefill the parameters for the convolution. 
//   conv_set_params(input, input_length, dilation, kernel, kernel_length);

//   // Start the convolution
//   start_conv();

//   // Read 
//   uint8_t status = conv_read_output(output, output_len);

//   // TODO: Handle status!
//   // if status
//   // if 
  
//   // volatile conv_status* curr_status = (volatile conv_status*) (STATUS_ADDR);

//   return get_register_status();
// }

// void perform_naive_convolution_1D(uint32_t *arr, size_t arr_len, uint32_t *kernel, size_t kernel_len, size_t dilation, float *output) {
//   size_t output_len = arr_len + (kernel_len - 1) * dilation;

//     for (int i = 0; i < output_len; i++) {
//         output[i] = 0.0f;

//         for (int j = 0; j < kernel_len; j++) {
//             int arr_index = i + j * dilation - (kernel_len - 1) * dilation;

//             uint32_t item = 0;
//             if (arr_index >= 0 && arr_index < arr_len) {
//                 item = arr[arr_index];
//             }

//             float float_input = *(float*)&item;
//             float float_kernel = *(float*)&kernel[j]; 
//             output[i] += float_input * float_kernel;
//         }
//     }
// }

// // NOTE: As of 12/9/25 the below get_register_status_human_readable() function cannot be included in drivers files because of the following:
// // 1) you cannot use printf in driver files.
// // 2) you cannot do general string manipulation.
// // The above will generate this relocation truncated to fit error
// // /bwrcq/C/louiejoshualabata/sp25-chips/software/baremetal-ide/platform/dsp25/src/hal_conv.c:200:(.text+0x5fa): relocation truncated to fit: R_RISCV_HI20 against `.LC0'
// // collect2: error: ld returned 1 exit status

// // NOTE: I believe this is due to how the sp25-baremetal-ide compiles .c into .riscv binaries because it is possible to do the above, (1) and (2). See /bwrcq/C/louiejoshualabata/sp25-chips/generators/dsp25-audio/baremetal_test/tests-dma/dma_utils.c and how this file is being included in this test /bwrcq/C/louiejoshualabata/sp25-chips/generators/dsp-1d-conv-sp25/baremetal_test/dma-temp.c


// // #include <string.h>
// // #include <stdio.h> // Required for sprintf

// // void get_register_status_human_readable(char* buffer) {

//   // if (buffer == NULL) {
//     //     return; // Exit if the pointer is NULL
//     // }
    
//     // // 1. CLEAR THE BUFFER
//     // buffer[0] = '\0'; 
    
//     // uint8_t status = get_register_status();
    
//     // // Pointer to the current end of the string, where new data will be written.
//     // // It starts pointing to the beginning of the buffer.
//     // char *write_ptr = buffer; 

//     // // 2. Check each flag using bitwise AND (&)
//     // // We use a temporary variable (len_written) to advance the write_ptr.
//     // int len_written = 0;

//     // if (status & STATUS_BUSY) {
//     //     len_written = sprintf(write_ptr, "BUSY | ");
//     //     write_ptr += len_written;
//     // }
//     // if (status & STATUS_COMPL) {
//     //     len_written = sprintf(write_ptr, "COMPL | ");
//     //     write_ptr += len_written;
//     // }
//     // if (status & STATUS_ERROR) {
//     //     len_written = sprintf(write_ptr, "ERROR | ");
//     //     write_ptr += len_written;
//     // }
//     // if (status & STATUS_INVALID) {
//     //     len_written = sprintf(write_ptr, "INVALID | ");
//     //     write_ptr += len_written;
//     // }
//     // if (status & STATUS_INFINITE) {
//     //     len_written = sprintf(write_ptr, "INFINITE | ");
//     //     write_ptr += len_written;
//     // }
//     // if (status & STATUS_OVERFLOW) {
//     //     len_written = sprintf(write_ptr, "OVERFLOW | ");
//     //     write_ptr += len_written;
//     // }
//     // if (status & STATUS_UNDERFLOW) {
//     //     len_written = sprintf(write_ptr, "UNDERFLOW | ");
//     //     write_ptr += len_written;
//     // }
//     // if (status & STATUS_INEXACT) {
//     //     len_written = sprintf(write_ptr, "INEXACT | ");
//     //     write_ptr += len_written;
//     // }
    
//     // // 3. Final string cleanup
    
//     // // If the buffer pointer hasn't moved (only contains the initial '\0' or is empty)
//     // if (write_ptr == buffer) { 
//     //     if (status == 0x00) {
//     //         sprintf(buffer, "IDLE/READY");
//     //     } else {
//     //         sprintf(buffer, "UNKNOWN STATUS");
//     //     }
//     // } else {
//     //     // Remove the trailing " | " 
//     //     // We know the length of the trailing part is 3 chars (" | ")
//     //     // We just move the write_ptr back 3 positions and null-terminate there.
//     //     write_ptr -= 3; 
//     //     *write_ptr = '\0';
//     // }
// // }