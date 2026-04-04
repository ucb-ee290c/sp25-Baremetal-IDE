#include "hal_conv.h"
#include "chip_config.h"

// --- 1D Convolution Driver Functions ---

void start_conv() {
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 1);
}

uint8_t get_register_status() {
  return reg_read8((uintptr_t)(MMIO_BASE + CONV_STATUS_ADDR));
}

uint32_t get_register_out_count() {
  return reg_read32((uintptr_t)(MMIO_BASE + CONV_COUNT_ADDR));
}

uint8_t get_register_read_check() {
  return reg_read8((uintptr_t)(MMIO_BASE + READ_CHECK_ADDR));
}

void conv_init() {
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  
  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 1);   
  reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 1);
}

// This function now ONLY writes parameters (Length, Dilation, Kernel)
int conv_set_params_kernel_only(uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length) {
  // Clear MMIO Reset, Clear Datapath, Stop
  reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 0);
  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 0); 
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  

  // Write parameters
  reg_write32((uintptr_t)(MMIO_BASE + CONV_LENGTH_ADDR), input_length);
  reg_write16((uintptr_t)(MMIO_BASE + CONV_DILATION_ADDR), dilation);

  // Write kernel data and kernel length encoding (One-time setup)
  if (kernel_length == 8) {
    for (int i = 0; i < 8; i += 2) {
      reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
    }
    reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 0);
  } else if (kernel_length == 16) {
    for (int i = 0; i < 16; i += 2) {
      reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
    }
    reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 1);  
  } else {
    return -1;  
  }

  return 0;
}

// Separate function to stream input data (used repeatedly in the main loop)
void conv_stream_input_batch(uint32_t *input, size_t start_element, size_t num_packets) {
    for (size_t i = 0; i < num_packets; ++i) {
        size_t element_idx = start_element + (i * FP32_PER_PACKET);
        reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR), *((uint64_t*) (input + element_idx)));
    }
}

// Separate function to read output data (used repeatedly in the main loop)
void conv_read_output_batch(uint32_t *output, size_t start_element, size_t num_packets) {
    for (size_t i = 0; i < num_packets; ++i) {
        size_t element_idx = start_element + (i * FP32_PER_PACKET);
        
        // Wait until at least one 64-bit word (2 FP32) is ready in the output FIFO
        while (get_register_out_count() < 1) {
            // Spin-wait for output data
        }
        
        uint64_t current_out = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
        
        // Unpack the 64-bit word into two 32-bit FP32 elements
        uint32_t *unpacked = (uint32_t *) &current_out;
        output[element_idx]   = unpacked[0];
        output[element_idx + 1] = unpacked[1];
    }
}

// The main streaming driver function
uint8_t perform_convolution_1D(
    uint32_t* input, 
    uint32_t input_length, 
    uint32_t* kernel, 
    uint8_t kernel_length, 
    uint32_t* output, 
    uint16_t dilation
) {
    // 1. Initial Setup
    conv_init();
    conv_set_params_kernel_only(input_length, dilation, kernel, kernel_length);

    // Calculate total packets
    size_t input_packets = input_length / FP32_PER_PACKET;
    // Assuming a valid convolution length, output_packets calculation is based on the DMA driver
    size_t kernel_packets = kernel_length / FP32_PER_PACKET; 
    size_t output_packets = input_packets + kernel_packets;

    size_t in_packet_idx = 0;
    size_t out_packet_idx = 0;

    // 2. Start the convolution engine
    start_conv();

    // 3. Streaming in Batches
    while (out_packet_idx < output_packets) {
        // Calculate batch size, capped by the FIFO capacity (8 packets)
        size_t remaining_out_packets = output_packets - out_packet_idx;

        // Pseudocode: current_batch_packets = min(FIFO_CAPACITY_PACKETS, remaining_out_packets);
        size_t current_batch_packets = remaining_out_packets < FIFO_CAPACITY_PACKETS
                                        ? remaining_out_packets
                                        : FIFO_CAPACITY_PACKETS;
        
        // The number of input packets to stream in this batch is limited by the remaining input
        // Pseudocode: input_stream_packets = min(current_batch_packets, remaining_in_packets)
        size_t input_stream_packets = 0;
        if (in_packet_idx < input_packets) {
            size_t remaining_in_packets = input_packets - in_packet_idx;
            input_stream_packets = remaining_in_packets < current_batch_packets
                                    ? remaining_in_packets
                                    : current_batch_packets;
        }

        // A. Stream Input Batch (MMIO write)
        if (input_stream_packets > 0) {
            // Convert packet index to FP32 element index
            size_t start_element = in_packet_idx * FP32_PER_PACKET; 
            conv_stream_input_batch(input, start_element, input_stream_packets);
            in_packet_idx += input_stream_packets;
        }

        // B. Read Output Batch (MMIO read)
        // Read the full batch, waiting for data if necessary (handled inside conv_read_output_batch)
        size_t start_element = out_packet_idx * FP32_PER_PACKET;
        conv_read_output_batch(output, start_element, current_batch_packets);
        out_packet_idx += current_batch_packets;

        // Note: Unlike the DMA driver which waits for bus silence, 
        // the MMIO driver uses the `get_register_out_count()` check 
        // inside `conv_read_output_batch` for synchronization (spin-wait).
    }

    // 4. Final Status Read (Output fully drained)
    return get_register_status();
}

// --- Utility 1D Convolution Driver Functions ---

void perform_naive_convolution_1D(uint32_t *arr, size_t arr_len, uint32_t *kernel, size_t kernel_len, size_t dilation, float *output) {
  size_t output_len = arr_len + (kernel_len - 1) * dilation;

    for (int i = 0; i < output_len; i++) {
        output[i] = 0.0f;

        for (int j = 0; j < kernel_len; j++) {
            int arr_index = i + j * dilation - (kernel_len - 1) * dilation;

            uint32_t item = 0;
            if (arr_index >= 0 && arr_index < arr_len) {
                item = arr[arr_index];
            }

            float float_input = *(float*)&item;
            float float_kernel = *(float*)&kernel[j]; 
            output[i] += float_input * float_kernel;
        }
    }
}

// NOTE: As of 12/9/25 the below get_register_status_human_readable() function cannot be included in drivers files because of the following:
// 1) you cannot use printf in driver files.
// 2) you cannot do general string manipulation.
// The above will generate this relocation truncated to fit error
// /bwrcq/C/louiejoshualabata/sp25-chips/software/baremetal-ide/platform/dsp25/src/hal_conv.c:200:(.text+0x5fa): relocation truncated to fit: R_RISCV_HI20 against `.LC0'
// collect2: error: ld returned 1 exit status

// NOTE: I believe this is due to how the sp25-baremetal-ide compiles .c into .riscv binaries because it is possible to do the above, (1) and (2). See /bwrcq/C/louiejoshualabata/sp25-chips/generators/dsp25-audio/baremetal_test/tests-dma/dma_utils.c and how this file is being included in this test /bwrcq/C/louiejoshualabata/sp25-chips/generators/dsp-1d-conv-sp25/baremetal_test/dma-temp.c


// #include <string.h>
// #include <stdio.h> // Required for sprintf

// void get_register_status_human_readable(char* buffer) {

  // if (buffer == NULL) {
    //     return; // Exit if the pointer is NULL
    // }
    
    // // 1. CLEAR THE BUFFER
    // buffer[0] = '\0'; 
    
    // uint8_t status = get_register_status();
    
    // // Pointer to the current end of the string, where new data will be written.
    // // It starts pointing to the beginning of the buffer.
    // char *write_ptr = buffer; 

    // // 2. Check each flag using bitwise AND (&)
    // // We use a temporary variable (len_written) to advance the write_ptr.
    // int len_written = 0;

    // if (status & STATUS_BUSY) {
    //     len_written = sprintf(write_ptr, "BUSY | ");
    //     write_ptr += len_written;
    // }
    // if (status & STATUS_COMPL) {
    //     len_written = sprintf(write_ptr, "COMPL | ");
    //     write_ptr += len_written;
    // }
    // if (status & STATUS_ERROR) {
    //     len_written = sprintf(write_ptr, "ERROR | ");
    //     write_ptr += len_written;
    // }
    // if (status & STATUS_INVALID) {
    //     len_written = sprintf(write_ptr, "INVALID | ");
    //     write_ptr += len_written;
    // }
    // if (status & STATUS_INFINITE) {
    //     len_written = sprintf(write_ptr, "INFINITE | ");
    //     write_ptr += len_written;
    // }
    // if (status & STATUS_OVERFLOW) {
    //     len_written = sprintf(write_ptr, "OVERFLOW | ");
    //     write_ptr += len_written;
    // }
    // if (status & STATUS_UNDERFLOW) {
    //     len_written = sprintf(write_ptr, "UNDERFLOW | ");
    //     write_ptr += len_written;
    // }
    // if (status & STATUS_INEXACT) {
    //     len_written = sprintf(write_ptr, "INEXACT | ");
    //     write_ptr += len_written;
    // }
    
    // // 3. Final string cleanup
    
    // // If the buffer pointer hasn't moved (only contains the initial '\0' or is empty)
    // if (write_ptr == buffer) { 
    //     if (status == 0x00) {
    //         sprintf(buffer, "IDLE/READY");
    //     } else {
    //         sprintf(buffer, "UNKNOWN STATUS");
    //     }
    // } else {
    //     // Remove the trailing " | " 
    //     // We know the length of the trailing part is 3 chars (" | ")
    //     // We just move the write_ptr back 3 positions and null-terminate there.
    //     write_ptr -= 3; 
    //     *write_ptr = '\0';
    // }
// }


void dma_1dConvDriver(
    uint32_t *input_buffer_ptr,
    uint32_t *output_buffer_ptr,
    uint32_t *kernel_buffer_ptr,
    size_t    total_elements,    // number of FP32 elements in input (assume even)
    size_t    kernel_elements,   // number of FP32 elements in kernel (assume even)
    uint16_t  dilation,
    uint32_t  base_id
) {
    // Packets = number of 64-bit beats (each beat carries 2 FP32s)
    size_t input_packets  = total_elements   / FP32_PER_PACKET;
    size_t kernel_packets = kernel_elements  / FP32_PER_PACKET;

    // Keep this as in your original hardware protocol
    size_t output_packets = input_packets + kernel_packets;

    // --- Reset / configure accelerator ---
    reg_write8(START_ADDR, 0);
    reg_write8(CLEAR_ADDR, 1);
    reg_write8(CLEAR_ADDR, 0);

    reg_write32(LENGTH_ADDR,  total_elements);
    reg_write16(DILATION_ADDR, dilation);

    // ================================
    // 1) DMA the kernel once
    // ================================
    dma_transaction_t k_trans = {
        .core            = DMA_KERNEL_CORE,
        .transaction_id  = base_id,
        .addr_r          = (uint64_t)(uintptr_t)kernel_buffer_ptr,  // from memory
        .addr_w          = KERNEL_ADDR,        // to kernel MMIO FIFO/port
        .inc_r           = DMA_WORD_INC,       // walk memory
        .inc_w           = 0,                  // fixed MMIO address
        .len             = kernel_packets,     // N x 64-bit beats
        .logw            = DMA_WORD_LOGW,      // 8 bytes
        .do_interrupt    = false,
        .do_address_gate = true
    };

    set_DMA_P(k_trans.core, k_trans, true);
    start_DMA(k_trans.core, k_trans.transaction_id, NULL);

    // Wait once for kernel DMA to complete
    dma_wait_till_inactive(DMA_IDLE_THRESHOLD);

    // ================================
    // 2) Enable convolution engine
    // ================================
    reg_write8(START_ADDR, 1);

    // ================================
    // 3) DMA *all* output + input in big chunks
    // ================================

    // 3a) Output: accelerator → memory (read from OUTPUT_ADDR)
    dma_transaction_t tx_out = {
        .core            = DMA_OUTPUT_CORE,
        .transaction_id  = base_id + 1,
        .addr_r          = OUTPUT_ADDR,              // from MMIO FIFO
        .addr_w          = (uint64_t)(uintptr_t)output_buffer_ptr,  // to memory
        .inc_r           = 0,                        // fixed MMIO address
        .inc_w           = DMA_WORD_INC,             // walk memory
        .len             = output_packets,           // ALL output beats
        .logw            = DMA_WORD_LOGW,
        .do_interrupt    = false,
        .do_address_gate = true
    };
    set_DMA_P(tx_out.core, tx_out, true);
    start_DMA(tx_out.core, tx_out.transaction_id, NULL);

    // 3b) Input: memory → accelerator (write to INPUT_ADDR)
    dma_transaction_t tx_in = {
        .core            = DMA_INPUT_CORE,
        .transaction_id  = base_id + 2,
        .addr_r          = (uint64_t)(uintptr_t)input_buffer_ptr,  // from memory
        .addr_w          = INPUT_ADDR,               // to MMIO FIFO
        .inc_r           = DMA_WORD_INC,             // walk memory
        .inc_w           = 0,                        // fixed MMIO address
        .len             = input_packets,            // ALL input beats
        .logw            = DMA_WORD_LOGW,
        .do_interrupt    = false,
        .do_address_gate = true
    };
    set_DMA_P(tx_in.core, tx_in, true);
    start_DMA(tx_in.core, tx_in.transaction_id, NULL);

    // ================================
    // 4) Wait once for everything to finish
    // ================================
    dma_wait_till_inactive(DMA_IDLE_THRESHOLD);
}
