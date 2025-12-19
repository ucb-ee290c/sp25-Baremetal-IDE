#include "main.h"

#include <vector>

#include "whisper.h"
#include "ggml-cpu.h"

volatile int wait = 1;

extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);

static void whisper_print_progress_callback(struct whisper_context *ctx, struct whisper_state *state, int progress, void *user_data) {
  printf("Progress: %d\n", progress);
}

void app_init() {

  // Loop through all of C++'s static initializers to initialize statically defined objects
  printf("Initializing libc\n");

  for (void (**p)(void) = __init_array_start; p != __init_array_end; p ++) {
    (*p)();
  }

  // Optionally wait for the debugger (`wait` must be changed to 0 in the debugger to continue)
  // while (wait) {}

  // Initialize the CPU backend
  printf("Initializing model\n");

  ggml_backend_reg_t cpu_backend = ggml_backend_cpu_reg();
  ggml_backend_register(cpu_backend);

  // Set Whisper's parameters
  struct whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = false;
  cparams.flash_attn = false; // What does this do?

  // Load the model into memory
  printf("Loading model\n");

  struct whisper_context * ctx = whisper_init_from_file_with_params("", cparams);

  // Load the audio. For initial testing we're just using a short section of silence
  // to make sure that the model runs to completion. This should be replaced with
  // actual audio data.
  // Alternately, the rest of this function can be moved into the app_main loop
  // and use I2S for live transcription. It is unknown whether the chip can keep up
  // with that in real time or not.
  std::vector<float> audio;
  
  for (int i = 0; i < 2000; i ++) {
    audio.push_back(0.0f);
  }

  // Run the audio through the model
  // Outputs will be printed automatically through whisper_print_progress_callback
  printf("Running inference\n");

  whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

  wparams.progress_callback = whisper_print_progress_callback;

  whisper_full_parallel(ctx, wparams, audio.data(), audio.size(), 1);

  printf("Done\n");
}

void app_main() {
  
}

int main(int argc, char **argv) {
  app_init();

  return 0; // Remove for anything live

  while (1) {
    app_main();
  }
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
   asm volatile ("wfi");
  }
}