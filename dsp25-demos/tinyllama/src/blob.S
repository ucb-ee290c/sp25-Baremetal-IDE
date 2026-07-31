/*
 * Bake the quantized model .bin and the tokenizer .bin directly into the ELF via `.incbin`.
 *
 * `.incbin` copies the raw file bytes straight into a section — this is the ONLY sane way to embed
 * a ~1 GB blob. A C-array header (like borai's weights.h) would explode the compiler: borai's
 * 260 KB model is already 1.5 MB of source text, so 1.1 GB would be ~5 GB of text and GCC would
 * OOM/hang. `.incbin` is a plain file copy done by the assembler, so build time/memory stay small.
 *
 * The file paths come from the build system as -D macros; this file is `.S` (capitalized) so the C
 * preprocessor runs first and STR() stringizes the bare path into the quoted form `.incbin` wants.
 * The blobs land in .rodata, which the linker places in DRAM (LOAD segment), so they travel inside
 * the ELF and are present in memory for both Spike (loads the ELF) and the FPGA (tsi loads the ELF).
 */

#define STR_(x) #x
#define STR(x)  STR_(x)

    .section .rodata
    .balign 64
    .global g_tinyllama_model
g_tinyllama_model:
    .incbin STR(TINYLLAMA_MODEL_BIN)
    .global g_tinyllama_model_end
g_tinyllama_model_end:

    .balign 64
    .global g_tinyllama_tokenizer
g_tinyllama_tokenizer:
    .incbin STR(TINYLLAMA_TOKENIZER_BIN)
    .global g_tinyllama_tokenizer_end
g_tinyllama_tokenizer_end:
