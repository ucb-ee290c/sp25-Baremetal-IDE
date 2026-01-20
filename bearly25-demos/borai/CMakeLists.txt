### Executables ###
add_compile_options(-g)
add_executable(borai
  fp32/src/main.c
  fp32/src/syscalls.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(borai PUBLIC fp32/include)

add_executable(boraiq
  int8/src/main.c
)

target_include_directories(boraiq PUBLIC int8/include)
target_include_directories(boraiq PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

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