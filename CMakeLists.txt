########################################################################################################################
# file:  CMakeLists.txt
#
# usage: 
#   Edit "VARIABLES"-section to suit project requirements.
#   Build instructions:
#     cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake
#     cmake --build ./build/ --target all
#
#   Building on Windows:
#     cmake -S ./ -B ./build/ -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake
#     cmake --build ./build/ --target all
#   Cleaning:
#     cmake --build ./build/ --target clean
#   Install:
#     cmake --build ./build/ --target install
########################################################################################################################
cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
project(baremetal-ide LANGUAGES ASM C CXX)


#################################
# Options
#################################

option(BAREMETAL_BUILD_RISCV        "Build for RISC-V platform"               ON  )
option(BAREMETAL_BUILD_X86          "Build for x86 platform"                  OFF )

option(CHIP                         "Build for a specific platform"           OFF )
option(LINKER                "Linker to store program on DRAM or FLASH"       OFF )

option(PROF_COV                     "Build with profiling and coverage"       OFF )
option(GCNO_ONLY                    "Only build gcno files"                   OFF )
option(USE_PGO                      "Build with profile guided optimization"  OFF )
option(OPT_INFO                     "Build with optimization information"     OFF )

option(USE_FREERTOS                 "Build with FreeRTOS"                     OFF )
option(ENABLE_RVV                   "Build for RVV targets"                   ON)
set(RVV_TYPE                        "" CACHE STRING "RVV mode: 0=No RVV, 1=RVV (rv64gcv_zfh), 2=RVV+Zvfh (rv64gcv_zfh_zvfh)")
set_property(CACHE RVV_TYPE PROPERTY STRINGS 0 1 2)
option(BUILD_VECNN                  "Build vec-nn library"                    OFF)
option(BUILD_MFCC_LIB               "Build mfcc-lib library"                  ON)
option(THREAD_LIB                   "Build thread-lib library"                OFF)
option(BMARK_LIB                    "Build bmark-lib library"                 OFF)

# Specify whether we are trying to run on the physical chip or in sims/spike
set(PLATFORM                        "CHIP" CACHE STRING "Runtime platform: CHIP or SIMS")

# Define the variable with a default empty value
set(AFDO_PATH "" CACHE STRING "Path to autofdo profile data")
#################################
# Toolchain Targets
#################################

set(CMAKE_EXECUTABLE_SUFFIX            ".elf")


#################################
# Flags
#################################

# CPU architecture
# RVV_TYPE takes precedence when provided:
#   0 -> no RVV
#   1 -> existing RVV config (default)
#   2 -> RVV + Zvfh (f16 vector capable toolchain/ISA required)
if (NOT RVV_TYPE STREQUAL "")
  set(_RVV_MODE "${RVV_TYPE}")
else()
  if (ENABLE_RVV)
    set(_RVV_MODE "1")
  else()
    set(_RVV_MODE "0")
  endif()
endif()

if (_RVV_MODE STREQUAL "0")
  set(ARCH "rv64imafd")
elseif (_RVV_MODE STREQUAL "1")
  set(ARCH "rv64gcv_zfh")
elseif (_RVV_MODE STREQUAL "2")
  set(ARCH "rv64gcv_zfh_zvfh")
else()
  message(FATAL_ERROR "Invalid RVV_TYPE='${_RVV_MODE}'. Use 0, 1, or 2.")
endif()

message(STATUS "RVV mode=${_RVV_MODE} ARCH=${ARCH}")
set(ABI                 "lp64d")
set(CMODEL              "medany")
set(ARCH_FLAGS          -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})

# spec
set(SPECS               "nosys.specs")
set(SPEC_FLAGS          -specs=${SPECS})


# convert CHIP to lowercase
string(TOLOWER ${CHIP} CHIP)
string(TOLOWER ${LINKER} LINKER)

# linker script
# HACK: ideally this should be handled by glossy, but currently i couldn't 
# figure out a way to propagate the LINKER_SCRIPT variable to the compile and link commands
if (NOT CHIP)
  message(STATUS "Chip not specified, using default configuration")
  set(CHIP "default")

  set(LINKER_SCRIPT       "${CMAKE_SOURCE_DIR}/glossy/glossy.ld")
else()
  set(LINKER_SCRIPT       "${CMAKE_SOURCE_DIR}/platform/${CHIP}/${CHIP}-${LINKER}.ld")
endif()

add_compile_options(-O1)
add_compile_options(-Wall -Wextra)
add_compile_options(-Wno-error=coverage-mismatch)

if (PROF_COV)
  message(STATUS "Building with profiling and coverage")
  add_compile_options(-fprofile-arcs -ftest-coverage)
endif()

if (GCNO_ONLY)
  message(STATUS "Building only gcno files")
  add_compile_options(-ftest-coverage)
endif()

if (USE_PGO)
  add_compile_options(-fprofile-use)
endif()

if (OPT_INFO)
  add_compile_options(-fopt-info)
endif()

# Check if the variable is not empty and add compiler options accordingly
if(NOT "${AFDO_PATH}" STREQUAL "")
    message(STATUS "Using AutoFDO profile from: ${AFDO_PATH}")
    add_compile_options(-fauto-profile=${AFDO_PATH})
endif()

# add_compile_options(-ffunction-sections -fdata-sections -fno-common -fno-builtin-printf -fno-pie)
# add_compile_options(-Wall -Wextra -Warray-bounds -Wno-unused-parameter -Wcast-qual)
add_compile_options(${ARCH_FLAGS})
add_compile_options(${SPEC_FLAGS})

add_link_options(-static)
# add_link_options(-ffunction-sections -fdata-sections -fno-common -fno-builtin-printf -fno-pie)
# add_link_options(-Wall -Wextra -Warray-bounds -Wno-unused-parameter -Wcast-qual)
add_link_options(-nostartfiles)
add_link_options(${ARCH_FLAGS})
add_link_options(${SPEC_FLAGS})
add_link_options(-T ${LINKER_SCRIPT})
# add_link_options(-rtlib=compiler-rt)
# add_link_options(-Wl,--rtlib-path=/scratch/iansseijelly/riscv-llvm-install/lib/linux/libclang_rt.builtins-riscv64.a)
if (PROF_COV)
  add_link_options(-lgcov)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/platform/${CHIP})  # moved to fix linker issue with DMA

#################################
# Dependencies
#################################

add_subdirectory(driver)
add_subdirectory(glossy)
add_subdirectory(lib)
add_subdirectory(examples)

add_subdirectory(bmark-lib)

if(BUILD_MFCC_LIB)
  add_subdirectory(mfcc-lib)
endif()

if(BUILD_VECNN)
  add_subdirectory(vec-nn)
endif()

if(THREAD_LIB)
  add_subdirectory(thread-lib)
endif()

if(BMARK_LIB)
  add_subdirectory(bmark-lib)
endif()

#################################
# Build Target Directories
#################################

add_subdirectory(app)
add_subdirectory(bearly25-tests)
add_subdirectory(bearly25-bmarks)
add_subdirectory(bearly25-demos)
add_subdirectory(dsp25-demos)
add_subdirectory(dsp25-tests)
add_subdirectory(c2c25-tests)
add_subdirectory(dsp25-bmarks)


#################################
# Library Installation
#################################

set(CMAKE_INSTALL_PREFIX "$ENV{RISCV}/riscv64-unknown-elf")

set_property(TARGET glossy PROPERTY PUBLIC_HEADER
  glossy/include/riscv.h
  glossy/include/riscv_encoding.h
  glossy/include/riscv_interrupt.h
)

set_property(TARGET glossy PROPERTY RESOURCE
  glossy/glossy.ld
  glossy/glossy.specs
)

install(TARGETS glossy
  LIBRARY DESTINATION "lib"
  PUBLIC_HEADER DESTINATION "include"
  RESOURCE DESTINATION "lib"
)

install(TARGETS htif
  LIBRARY DESTINATION "lib"
  PUBLIC_HEADER DESTINATION "include"
  RESOURCE DESTINATION "lib"
)
