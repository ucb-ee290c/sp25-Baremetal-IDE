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
)

target_include_directories(boraiq-test PUBLIC int8/include)
target_include_directories(boraiq-test PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

if(BORAI_TESTQ_TRANSPOSED_WEIGHTS)
  target_compile_definitions(boraiq-test PRIVATE TRANSPOSED_WEIGHTS)
  target_include_directories(boraiq-test PRIVATE ${CMAKE_SOURCE_DIR}/vec-nn/include)
  if(BUILD_VECNN)
    target_link_libraries(boraiq-test PRIVATE vecnn)
  endif()
endif()

option(BORAI_TESTQ_PREFILL_MULTICORE
  "Enable dual-hart multicore prefill in boraiq-test (splits attention heads and FFN rows)" ON)

if(BORAI_TESTQ_PREFILL_MULTICORE)
  if (NOT TARGET threadlib)
    message(FATAL_ERROR "boraiq-test PREFILL_MULTICORE requires THREAD_LIB=ON (threadlib target missing).")
  endif()
  target_compile_definitions(boraiq-test PRIVATE PREFILL_MULTICORE)
  target_include_directories(boraiq-test PRIVATE ${CMAKE_SOURCE_DIR}/thread-lib)
  target_link_libraries(boraiq-test PRIVATE threadlib)
endif()

option(BORAI_TESTQ_VEC_SOFTMAX
  "Use vectorized RVV softmax for attention in boraiq-test (requires vecnn)" OFF)

if(BORAI_TESTQ_VEC_SOFTMAX)
  target_compile_definitions(boraiq-test PRIVATE VEC_SOFTMAX)
  target_include_directories(boraiq-test PRIVATE ${CMAKE_SOURCE_DIR}/vec-nn/include)
  if(BUILD_VECNN)
    target_link_libraries(boraiq-test PRIVATE vecnn)
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
