### Executables ###
add_compile_options(-g)
add_executable(borai-test
  fp32/src/main.c
  fp32/src/syscalls.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(borai-test PUBLIC fp32/include)

option(BORAI_TESTQ_TRANSPOSED_WEIGHTS
  "Enable transposed-weight vectorized matmul in boraiq-test (requires vecnn)" ON)

add_executable(boraiq-test
  int8/src/main.c
  int8/src/tiny_gemm_i8_rvv.c
  int8/src/tiny_vec_ops_rvv.c
)

target_include_directories(boraiq-test PUBLIC int8/include)

option(BORAI_TESTQ_MAX_PERF
  "Compile boraiq-test with aggressive per-target optimization flags" ON)

if(BORAI_TESTQ_MAX_PERF)
  target_compile_options(boraiq-test PRIVATE -O3 -fno-math-errno -fno-trapping-math)
endif()

option(BORAI_TESTQ_TINY_SHAPE_GEMM
  "Enable boraiq-test specialized RVV tiny-shape int8 matmul kernels (K=64/172)" ON)

if(BORAI_TESTQ_TINY_SHAPE_GEMM)
  target_compile_definitions(boraiq-test PRIVATE BORAIQ_TINY_SHAPE_GEMM)
endif()

option(BORAI_TESTQ_FAST_SAMPLER_512
  "Enable boraiq-test specialized sampler path for vocab_size=512" ON)

if(BORAI_TESTQ_FAST_SAMPLER_512)
  target_compile_definitions(boraiq-test PRIVATE BORAIQ_FAST_SAMPLER_512)
endif()

option(BORAI_TESTQ_BENCH_SUPPRESS_TOKEN_IO
  "Suppress per-token decode/print in very short benchmarks to avoid UART overhead" ON)

if(BORAI_TESTQ_BENCH_SUPPRESS_TOKEN_IO)
  target_compile_definitions(
    boraiq-test PRIVATE
    BORAIQ_BENCH_SUPPRESS_TOKEN_IO
    BORAIQ_BENCH_STREAM_TOKENS_MIN_STEPS=8)
endif()

option(BORAI_TESTQ_BENCH_NO_TOKEN_FLUSH
  "Disable per-token fflush(stdout) in generation path" ON)

if(BORAI_TESTQ_BENCH_NO_TOKEN_FLUSH)
  target_compile_definitions(boraiq-test PRIVATE BORAIQ_BENCH_NO_TOKEN_FLUSH)
endif()

option(BORAI_TESTQ_TINY_VEC_OPS
  "Enable boraiq-test specialized tiny-shape quantize/rmsnorm kernels (n=64/172)" ON)

if(BORAI_TESTQ_TINY_VEC_OPS)
  target_compile_definitions(boraiq-test PRIVATE BORAIQ_TINY_VEC_OPS)
endif()

option(BORAI_TESTQ_TINY_ATTN_H8
  "Enable boraiq-test specialized tiny attention kernels for head_size=8" ON)

if(BORAI_TESTQ_TINY_ATTN_H8)
  target_compile_definitions(boraiq-test PRIVATE BORAIQ_TINY_ATTN_H8)
endif()

option(BORAI_TESTQ_FAST_SWIGLU_EXP
  "Enable boraiq-test fast SwiGLU sigmoid approximation (replaces expf in SiLU)" ON)

if(BORAI_TESTQ_FAST_SWIGLU_EXP)
  target_compile_definitions(boraiq-test PRIVATE BORAIQ_FAST_SWIGLU_EXP)
endif()

if(BORAI_TESTQ_TRANSPOSED_WEIGHTS)
  if(BUILD_VECNN)
    target_compile_definitions(boraiq-test PRIVATE TRANSPOSED_WEIGHTS)
    target_include_directories(boraiq-test PRIVATE ${CMAKE_SOURCE_DIR}/vec-nn/include)
    target_link_libraries(boraiq-test PRIVATE vecnn)
  else()
    message(STATUS "boraiq-test: BUILD_VECNN is OFF, disabling TRANSPOSED_WEIGHTS.")
  endif()
endif()

option(BORAI_TESTQ_PREFILL_MULTICORE
  "Enable dual-hart multicore prefill in boraiq-test (splits attention heads and FFN rows)" ON)

option(BORAI_TESTQ_MC_REBALANCE_QKV
  "Rebalance multicore QKV stage: hart0=wq, hart1=wk+wv (for dim=64, kv_dim=32)" ON)

if(BORAI_TESTQ_PREFILL_MULTICORE)
  if (NOT TARGET threadlib)
    message(STATUS "boraiq-test: THREAD_LIB is OFF, disabling PREFILL_MULTICORE (single-core fallback).")
  else()
    target_compile_definitions(boraiq-test PRIVATE PREFILL_MULTICORE)
    if(BORAI_TESTQ_MC_REBALANCE_QKV)
      target_compile_definitions(boraiq-test PRIVATE BORAIQ_MC_REBALANCE_QKV)
    endif()
    target_include_directories(boraiq-test BEFORE PRIVATE ${CMAKE_SOURCE_DIR})
    target_include_directories(boraiq-test BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/thread-lib)
    target_link_libraries(boraiq-test PRIVATE threadlib)
  endif()
endif()

option(BORAI_TESTQ_VEC_SOFTMAX
  "Use vectorized RVV softmax for attention in boraiq-test (requires vecnn)" ON)

if(BORAI_TESTQ_VEC_SOFTMAX)
  if(BUILD_VECNN)
    target_compile_definitions(boraiq-test PRIVATE VEC_SOFTMAX)
    target_include_directories(boraiq-test PRIVATE ${CMAKE_SOURCE_DIR}/vec-nn/include)
    target_link_libraries(boraiq-test PRIVATE vecnn)
  else()
    message(STATUS "boraiq-test: BUILD_VECNN is OFF, disabling VEC_SOFTMAX.")
  endif()
endif()

### Dependencies ###

if(BUILD_VECNN)
  target_link_libraries(borai-test PRIVATE vecnn)
endif()

target_link_libraries(borai-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
## Include Math.h library
target_link_libraries(borai-test PRIVATE m)

target_link_libraries(boraiq-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
## Include Math.h library
target_link_libraries(boraiq-test PRIVATE m)

target_include_directories(borai-test PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)
