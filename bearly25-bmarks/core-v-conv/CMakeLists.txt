# CMakeLists definitions for target `core-v-conv`.

#################################
# Build Configuration
#################################

# Source Files
add_executable(core-v-conv
  src/main.c
  src/bench_fill.c
  src/bench_impl.c
  src/bench_sizes.c
)

# Header Files
target_include_directories(core-v-conv PUBLIC include)

#################################
# Dependencies
#################################

target_link_libraries(core-v-conv PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(BUILD_VECNN)
  target_link_libraries(core-v-conv PRIVATE vecnn)
  target_compile_definitions(core-v-conv PRIVATE BENCH_HAS_VECNN=1)
endif()

if (PROF_COV)
  target_link_libraries(core-v-conv PRIVATE gcov)
endif()
