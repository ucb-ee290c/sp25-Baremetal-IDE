# tinyllama: llama2.c runq (grouped Q8_0 int8) ported to dsp25 baremetal.
# Model + tokenizer are PRELOADED into DRAM at fixed addresses (see include/tinyllama_config.h);
# nothing is embedded in the ELF. Needs the 2 GiB / big-heap linker variant, so build with LINKER=llm:
#
#   make build CHIP=dsp25 PLATFORM=CHIP LINKER=llm TARGET=tinyllama
#
# (LINKER=llm -> platform/dsp25/dsp25-llm.ld, wired in the top-level CMakeLists.)

# Paths to the quantized model + tokenizer that get baked into the ELF via .incbin (src/blob.S).
# Produce the model with:  python llama2.c/export.py <out> --version 2 --checkpoint <tinyllama>
# (version 2 = grouped Q8_0, what this runtime expects). Drop both files at these paths, or override
# with -DTINYLLAMA_MODEL_BIN=/abs/path.bin -DTINYLLAMA_TOKENIZER_BIN=/abs/path.bin.
set(TINYLLAMA_MODEL_BIN     "${CMAKE_CURRENT_SOURCE_DIR}/model/model_q80.bin"  CACHE STRING "Path to the version-2 grouped-Q8_0 model .bin to embed")
set(TINYLLAMA_TOKENIZER_BIN "${CMAKE_CURRENT_SOURCE_DIR}/model/tokenizer.bin"  CACHE STRING "Path to the tokenizer .bin to embed")
if(NOT EXISTS "${TINYLLAMA_MODEL_BIN}")
  message(WARNING "tinyllama: model blob not found at ${TINYLLAMA_MODEL_BIN} — place it there (or set -DTINYLLAMA_MODEL_BIN) before building; the .incbin will otherwise fail.")
endif()

add_executable(tinyllama
  src/main.c
  src/blob.S                                    # .incbin of the model + tokenizer
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(tinyllama PRIVATE
  include
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/driver/rocket-chip-blocks/uart
)

# -O2 for the matmul-heavy forward pass.
target_compile_options(tinyllama PRIVATE -O2)

# Bake-in mode + blob paths (bare paths; blob.S stringizes them for .incbin).
target_compile_definitions(tinyllama PRIVATE
  TINYLLAMA_EMBED=1
  TINYLLAMA_MODEL_BIN=${TINYLLAMA_MODEL_BIN}
  TINYLLAMA_TOKENIZER_BIN=${TINYLLAMA_TOKENIZER_BIN}
)

# Optional compile-time prompt for non-interactive/Spike autorun (empty => interactive UART loop).
# e.g. EXTRA_CMAKE_ARGS="-DTINYLLAMA_PROMPT=Once upon a time -DTINYLLAMA_TEMPERATURE=0"
set(TINYLLAMA_PROMPT "" CACHE STRING "Compile-time prompt; empty = interactive")
if(NOT TINYLLAMA_PROMPT STREQUAL "")
  target_compile_definitions(tinyllama PRIVATE TINYLLAMA_PROMPT=\"${TINYLLAMA_PROMPT}\")
endif()
if(DEFINED TINYLLAMA_TEMPERATURE)
  target_compile_definitions(tinyllama PRIVATE TINYLLAMA_TEMPERATURE=${TINYLLAMA_TEMPERATURE})
endif()
if(DEFINED TINYLLAMA_STEPS)
  target_compile_definitions(tinyllama PRIVATE TINYLLAMA_STEPS=${TINYLLAMA_STEPS})
endif()

target_link_libraries(tinyllama PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if (PROF_COV)
  target_link_libraries(tinyllama PRIVATE gcov)
endif()
