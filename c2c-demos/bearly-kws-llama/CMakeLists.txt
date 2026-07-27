# bearly-kws-llama: dual-core combined demo (Bearly side).
#   hart 0: C2C KWS receiver (reuses c2c-demos/bearly-kws-rolling/src/main.c) + keyword controller.
#   hart 1: TinyLlama int8 inference (reuses bearly25-demos/borai/int8/src/main.c).
# A confident keyword (yes/go/on) starts Llama on hart 1; (no/off/stop) aborts + halts it.
#
# Build (Bearly side), needs vecnn for the borai transposed-weight matmul:
#   make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-kws-llama EXTRA_CMAKE_ARGS="-DBUILD_VECNN=ON"
#
# Both source units are compiled with guard macros that are inert in their standalone builds
# (bearly-kws-rolling / boraiq), so this target does not change those demos.

if(NOT TARGET vecnn)
  message(STATUS "bearly-kws-llama: skipped (requires vecnn; configure with -DBUILD_VECNN=ON on CHIP=bearly25).")
  return()
endif()

set(KWS_ROLLING_DIR ${CMAKE_SOURCE_DIR}/c2c-demos/bearly-kws-rolling)
set(BORAI_INT8_DIR  ${CMAKE_SOURCE_DIR}/bearly25-demos/borai/int8)
set(TINYSPEECH_DIR  ${CMAKE_SOURCE_DIR}/bearly25-bmarks/tinyspeech-mc)

add_executable(bearly-kws-llama
  ${KWS_ROLLING_DIR}/src/main.c                 # hart 0: KWS receiver + controller (guarded hooks)
  ${BORAI_INT8_DIR}/src/main.c                  # hart 1: Llama int8 inference (guarded entry points)
  ${CMAKE_SOURCE_DIR}/c2c-demos/common/c2c_shm.c
  ${TINYSPEECH_DIR}/src/tinyspeech_model.c
  ${TINYSPEECH_DIR}/src/tinyspeech_int8.c
  ${TINYSPEECH_DIR}/src/tensor.c
  ${TINYSPEECH_DIR}/src/misc.c
  ${TINYSPEECH_DIR}/src/modules.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(bearly-kws-llama PRIVATE
  ${KWS_ROLLING_DIR}/include        # KWS "main.h" (bearly-kws-rolling) for that TU
  ${CMAKE_SOURCE_DIR}/c2c-demos/common
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${TINYSPEECH_DIR}/include         # tinyspeech weights.h/main.h — must precede borai's same-named headers
  ${CMAKE_SOURCE_DIR}/vec-nn/include
  ${BORAI_INT8_DIR}/include         # LAST: borai reaches its own headers via the local-dir "borai_main.h" wrapper,
                                    # so ordering it last avoids its weights.h/main.h shadowing tinyspeech's.
)

target_compile_options(bearly-kws-llama PRIVATE -O2)

target_compile_definitions(bearly-kws-llama PRIVATE
  # --- combined-demo integration guards ---
  KWS_BEARLY_LLAMA=1          # KWS TU: dual-core controller + hart-1 entry + malloc lock
  KWS_LLAMA_COMBINED=1        # borai TU: neutralize its main/__main/app_main/target_frequency/softmax
  TRANSPOSED_WEIGHTS          # borai: vectorized transposed-weight matmul (needs vecnn)
  # --- KWS side: single-core float TinySpeech pipeline (proven-correct path) ---
  KWS_BEARLY_ROLLING_USE_THREADLIB=0
  TINYSPEECH_OPT_VARIANT_GEMM_RVV=1
  TINYSPEECH_CONV_FUSE_RELU=1
  TINYSPEECH_FUSE_POOL=1
  TINYSPEECH_OUTPUT_SOFTMAX=0
  TINYSPEECH_ENABLE_TRACE=0
  TINYSPEECH_REF_CHECK_STAGE_SUM=0
  TINYSPEECH_USE_TENSOR_ARENA=1
  TINYSPEECH_TENSOR_ARENA_SIZE=524288
  TINYSPEECH_FP16_MIXED=0
  TINYSPEECH_FP16_FC_ACCEL=0
  TINYSPEECH_INT8_PIPELINE=0
  TINYSPEECH_INT8_USE_VSSE8_STORE=0
  TINYSPEECH_INT8_USE_VSE8_PACK_STORE=0
  TINYSPEECH_RVV_NCLIP_API=0
  TINYSPEECH_INT8_USE_VSMUL_REQUANT=0
  TINYSPEECH_RVV_VSMUL_API=0
  TINYSPEECH_INT8_RVV_UKERNELS=1
  TINYSPEECH_INT8_MULTICORE=0
  # live audio has no ground truth; INPUT-CMP against the reference is meaningless here
  KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE=0
)

target_link_libraries(bearly-kws-llama PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config
  rocketcore
  vecnn
  m
)

# Heap: Llama runstate/kv-cache + transposed weights (~2 MB) plus TinySpeech per-case tensors.
target_link_options(bearly-kws-llama PRIVATE -Wl,--defsym,__heap_size=8388608)

if(PROF_COV)
  target_link_libraries(bearly-kws-llama PRIVATE gcov)
endif()
