# CMakeLists definitions for target `memory-latency`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(memory-latency
  src/main.c
  src/memlat_core.c
  src/memlat_l1_l2_dram.c
  src/memlat_noc_bg.c
  src/memlat_patterns.c
  src/memlat_scratch_tcm.c
)

# Header Files
target_include_directories(memory-latency PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(memory-latency PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
  target_link_libraries(memory-latency PRIVATE gcov)
endif()
