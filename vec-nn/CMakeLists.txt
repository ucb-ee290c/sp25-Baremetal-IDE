file(GLOB_RECURSE VECNN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_library(vecnn STATIC ${VECNN_SRC})

target_include_directories(vecnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(vecnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

option(VECNN_MAX_PERF
  "Compile vecnn kernels with aggressive optimization flags" ON)

if(VECNN_MAX_PERF)
  target_compile_options(vecnn PRIVATE -O3 -funroll-loops -fno-math-errno -fno-trapping-math)
endif()
