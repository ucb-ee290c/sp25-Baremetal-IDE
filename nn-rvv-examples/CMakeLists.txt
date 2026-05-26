# nn-rvv-examples: reference MNIST inference programs that exercise the
# nn-rvv kernel library end-to-end. Each subdir builds a small executable
# that links nnrvv (kernels) + glossy (bare-metal runtime + HTIF) and runs
# on Spike / RTL sim under whatever CHIP platform the parent build picked.
#
# Threading: examples call nn_rvv_threading_init() in app_main and use the
# library kernels directly, so the parallelism level is set at configure
# time via -DNN_RVV_N_HARTS=<n>. NN_RVV_N_HARTS=1 runs everything serially.

if(NOT BUILD_NN_RVV)
  message(FATAL_ERROR
    "nn-rvv-examples requires the nn-rvv library. Configure with "
    "-DBUILD_NN_RVV=ON.")
endif()

add_subdirectory(mnist)
add_subdirectory(mnist_cnn)
add_subdirectory(mnist_cnn_quant)
add_subdirectory(mnist_quant)
add_subdirectory(mnist_quant2)
