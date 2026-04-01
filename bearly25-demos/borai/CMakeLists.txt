### Executables ###
add_compile_options(-g)
add_executable(borai
  fp32/src/main.c
  fp32/src/syscalls.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(borai PUBLIC fp32/include)

option(BORAIQ_TRANSPOSED_WEIGHTS
  "Enable transposed-weight vectorized matmul in boraiq (requires vecnn)" ON)

add_executable(boraiq
  int8/src/main.c
)

target_include_directories(boraiq PUBLIC int8/include)
target_include_directories(boraiq PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

if(BORAIQ_TRANSPOSED_WEIGHTS)
  target_compile_definitions(boraiq PRIVATE TRANSPOSED_WEIGHTS)
  target_include_directories(boraiq PRIVATE ${CMAKE_SOURCE_DIR}/vec-nn/include)
  if(BUILD_VECNN)
    target_link_libraries(boraiq PRIVATE vecnn)
  endif()
endif()

option(BORAIQ_PREFILL_MULTICORE
  "Enable dual-hart multicore prefill in boraiq (splits attention heads and FFN rows)" ON)

if(BORAIQ_PREFILL_MULTICORE)
  target_compile_definitions(boraiq PRIVATE PREFILL_MULTICORE)
endif()

option(BORAIQ_VEC_SOFTMAX
  "Use vectorized RVV softmax for attention in boraiq (requires vecnn)" OFF)

if(BORAIQ_VEC_SOFTMAX)
  target_compile_definitions(boraiq PRIVATE VEC_SOFTMAX)
  target_include_directories(boraiq PRIVATE ${CMAKE_SOURCE_DIR}/vec-nn/include)
  if(BUILD_VECNN)
    target_link_libraries(boraiq PRIVATE vecnn)
  endif()
endif()

### Dependencies ###

if(BUILD_VECNN)
  target_link_libraries(borai PRIVATE vecnn)
endif()

target_link_libraries(borai PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
## Include Math.h library
target_link_libraries(borai PRIVATE m)

target_link_libraries(boraiq PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)
## Include Math.h library
target_link_libraries(boraiq PRIVATE m)

target_include_directories(borai PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)