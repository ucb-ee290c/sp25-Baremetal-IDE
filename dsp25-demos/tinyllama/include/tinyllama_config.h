#ifndef TINYLLAMA_CONFIG_H
#define TINYLLAMA_CONFIG_H

/* ------------------------------------------------------------------------------------------------
 * dsp25 baremetal TinyLlama (llama2.c runq int8, grouped Q8_0) — tunables + memory map.
 *
 * There is NO filesystem on the target. Preload two blobs into DRAM out-of-band (however your
 * loader does it) BEFORE running:
 *   - the model     : `python llama2.c/export.py model_q80.bin --version 2 --checkpoint <tinyllama.pt>`
 *                     placed at TINYLLAMA_WEIGHTS_BASE
 *   - the tokenizer : llama2.c tokenizer.bin (Llama 32000 SentencePiece) at TINYLLAMA_TOKENIZER_BASE
 * The code reads Config + group_size straight from the model .bin's 256-byte v2 header, so it
 * adapts to whatever model you load (TinyLlama-1.1B, stories, etc.).
 *
 * Memory map (DRAM ORIGIN = 0x8000_0000 = 2 GiB; you have 8 GiB => valid to ~0x2_8000_0000 = 10 GiB):
 *   [0x0_8000_0000, 0x1_0000_0000)  program: .text/.data/.bss + heap + stack  (linker dsp25-llm.ld, 2 GiB)
 *   [0x1_0000_0000, 0x1_8000_0000)  model .bin blob  (WEIGHTS_BASE, up to 2 GiB — TinyLlama Q8 ~1.1 GiB)
 *    0x1_8000_0000                  tokenizer.bin blob (TOKENIZER_BASE, ~1 MiB)
 * The linker never touches anything at/above WEIGHTS_BASE, so the preloaded blobs are safe.
 * ---------------------------------------------------------------------------------------------- */

/* Two ways to get the model + tokenizer into memory:
 *  - TINYLLAMA_EMBED (default): the blobs are baked INTO the ELF via .incbin (see src/blob.S) and
 *    the loader points at those symbols. Weights travel inside the ELF — no separate preload; works
 *    for both Spike (loads the ELF) and the FPGA (tsi loads the ELF). ELF grows by the model size.
 *  - else: the blobs are preloaded out-of-band into DRAM at the fixed absolute addresses below, and
 *    nothing is embedded (tiny ELF). Use this if you build a fast DRAM-preload path.
 */
#ifndef TINYLLAMA_EMBED
#define TINYLLAMA_EMBED 1
#endif

#if TINYLLAMA_EMBED
extern const unsigned char g_tinyllama_model[];      /* .incbin'd model .bin (src/blob.S) */
extern const unsigned char g_tinyllama_tokenizer[];  /* .incbin'd tokenizer .bin */
#define TINYLLAMA_WEIGHTS_BASE   ((uintptr_t)g_tinyllama_model)
#define TINYLLAMA_TOKENIZER_BASE ((uintptr_t)g_tinyllama_tokenizer)
#else
#ifndef TINYLLAMA_WEIGHTS_BASE
#define TINYLLAMA_WEIGHTS_BASE   0x100000000ULL   /* 4 GiB absolute (2 GiB above the program region) */
#endif
#ifndef TINYLLAMA_TOKENIZER_BASE
#define TINYLLAMA_TOKENIZER_BASE 0x180000000ULL   /* 6 GiB absolute (2 GiB slot for the model below) */
#endif
#endif

/* Operating frequency (PLL target via init_test). Higher = faster tokens; keep it to a value the
 * DSP is stable at. */
#ifndef TINYLLAMA_TARGET_FREQUENCY_HZ
#define TINYLLAMA_TARGET_FREQUENCY_HZ 500000000ULL
#endif

/* Sampling: 0.0 = greedy; 1.0 = original. top-p 0.9 works well. */
#ifndef TINYLLAMA_TEMPERATURE
#define TINYLLAMA_TEMPERATURE 1.0f
#endif
#ifndef TINYLLAMA_TOPP
#define TINYLLAMA_TOPP 0.9f
#endif

/* Tokens generated per prompt (0 or > seq_len => clamped to the model's seq_len). */
#ifndef TINYLLAMA_STEPS
#define TINYLLAMA_STEPS 256
#endif

/* Max serial prompt length (bytes). */
#ifndef TINYLLAMA_PROMPT_MAX
#define TINYLLAMA_PROMPT_MAX 512
#endif

/* Non-interactive autorun prompt. Spike has no UART input, so for a Spike run define a compile-time
 * prompt here: app_main generates from it once, then halts (instead of the interactive UART loop
 * used on the FPGA). For a deterministic token stream you can diff against host `runq`, also set
 * TINYLLAMA_TEMPERATURE to 0.0f. Uncomment/edit:
 *   #define TINYLLAMA_PROMPT "Once upon a time"
 * (Leave undefined for the interactive UART prompt loop.) */
/* #define TINYLLAMA_PROMPT "Once upon a time" */

#endif /* TINYLLAMA_CONFIG_H */
