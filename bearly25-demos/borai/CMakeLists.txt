### borai (fp32) + boraiq (int8) — both fully depend on nn-rvv.
###
### All kernels (matmul, softmax, rmsnorm, RoPE, transpose, attention) live in
### nn-rvv. The targets here only contain model-specific glue + driver code.

if(NOT BUILD_NN_RVV)
  message(FATAL_ERROR
    "borai/boraiq require nn-rvv. Re-configure with -DBUILD_NN_RVV=ON "
    "(and optionally -DNN_RVV_N_HARTS=<n> for multi-hart parallelism).")
endif()

add_compile_options(-g)

### borai (fp32) ###

add_executable(borai
  fp32/src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# borai's local syscalls.c routes _write through UART0. On the chip / RTL sim
# that's correct, but on Spike there is no UART and main() leaves it
# uninitialized anyway, so printf would silently vanish. When PLATFORM=SIMS,
# omit the override and fall through to glossy's HTIF-based syscalls.
if(NOT PLATFORM_LOWER STREQUAL "sims")
  target_sources(borai PRIVATE fp32/src/syscalls.c)
endif()

target_include_directories(borai PUBLIC  fp32/include)
target_include_directories(borai PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)
target_link_libraries(borai PRIVATE nnrvv m)
target_link_libraries(borai PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

### boraiq (int8) ###

add_executable(boraiq
  int8/src/main.c
)

target_include_directories(boraiq PUBLIC  int8/include)
target_include_directories(boraiq PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)
target_link_libraries(boraiq PRIVATE nnrvv m)
target_link_libraries(boraiq PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

### Heap sizing.
###
### boraiq allocates a transposed copy of every weight matrix on top of the
### RunState buffers (KV cache, attention scratch, logits, etc.); 16 MiB
### covers small-LLaMA-2 sized models comfortably.
### borai (fp32) is ~4x the boraiq memory pressure; 32 MiB is safe headroom.
### Override at configure time:
###   cmake ... -DBORAI_HEAP_SIZE=67108864 -DBORAIQ_HEAP_SIZE=33554432
set(BORAIQ_HEAP_SIZE "16777216" CACHE STRING "Heap size (bytes) for boraiq (int8) target")
set(BORAI_HEAP_SIZE  "33554432" CACHE STRING "Heap size (bytes) for borai (fp32) target")

target_link_options(boraiq PRIVATE -Wl,--defsym,__heap_size=${BORAIQ_HEAP_SIZE})
target_link_options(borai  PRIVATE -Wl,--defsym,__heap_size=${BORAI_HEAP_SIZE})
