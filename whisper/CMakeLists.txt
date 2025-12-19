# CMakeLists definitions for target `app`.

#################################
# Build Configuration
#################################

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED)

# Configure warnings: some warnings are signs of really bad things here, so error except on known safe warnings
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
add_compile_options(
  -Wno-error=unused-variable
  -Wno-error=unused-parameter
  -Wno-error=unused-function
  -Wno-error=cast-qual
  -Wno-error=maybe-uninitialized
)
# I can't figure out how to do the same for the linker... assume most linker warnings are bad

# Disable dynamic libraries
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared libraries" FORCE)

# Add fake threads stub
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Configure whisper.cpp
set(GGML_OPENMP OFF)

add_compile_definitions(BAREMETAL)

# -mcmodel-medany is needed to appease the linker. This also requires a custom riscv64-unknown-elf build
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=medany -Wno-cast-qual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medany -Wno-cast-qual")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcmodel=medany")
add_link_options(-mcmodel=medany)

# Source Files

add_executable(whisper_dsp
  src/main.cpp
)

# Turn the model into a C data file
set(WHISPER_MODEL_BIN "${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp/models/ggml-tiny.en.bin")
set(WHISPER_MODEL_C "${CMAKE_CURRENT_SOURCE_DIR}/whisper_overrides/whisper_model_data.c")
set(WHISPER_MODEL_PY_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/file_to_c_data.py")

add_custom_command(
  OUTPUT ${WHISPER_MODEL_C}
  COMMAND ${Python3_EXECUTABLE} ${WHISPER_MODEL_PY_SCRIPT} ${WHISPER_MODEL_BIN} ${WHISPER_MODEL_C}
  DEPENDS ${WHISPER_MODEL_PY_SCRIPT} ${WHISPER_MODEL_BIN}
  VERBATIM
)

add_custom_target(whisper_model ALL DEPENDS ${WHISPER_MODEL_C})

add_dependencies(whisper_dsp whisper_model)

# Add overrides for whisper functions
target_sources(whisper_dsp PRIVATE
  whisper_overrides/time.c
  whisper_overrides/model.cpp
  whisper_overrides/whisper_model_data.c
)

# Add build dependencies
add_subdirectory(whisper.cpp/src)
add_subdirectory(whisper.cpp/ggml)


# Header Files
target_include_directories(whisper_dsp PUBLIC include)
target_include_directories(whisper_dsp PUBLIC whisper.cpp/include)
target_include_directories(whisper_dsp PUBLIC whisper.cpp/ggml/include)

#################################
# Dependencies
#################################

target_link_libraries(whisper_dsp PRIVATE 
-L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if (PROF_COV)
target_link_libraries(whisper_dsp PRIVATE gcov)
endif()

target_link_libraries(whisper_dsp PRIVATE whisper ggml)
