# CMakeLists definitions for target `memory-latency`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(memory-latency
  src/main.c
  src/memlat_core.c
  src/memlat_l1_l2_dram.c
  src/memlat_scratch_tcm.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# Header Files
target_include_directories(memory-latency PUBLIC include)
target_include_directories(memory-latency PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

#################################
# Dependencies
#################################

target_link_libraries(memory-latency PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(memory-latency PRIVATE gcov)
endif()
