########################################################################################################################
# Baremetal Platform Configuration for BearlyML'25
########################################################################################################################

file(GLOB_RECURSE PLATFORM_FILES 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c )
add_library(chip-config STATIC chip.c ${PLATFORM_FILES})

# set terminal device (printf, scanf, etc.) to HTIF
set(TERMINAL_DEVICE_HTIF    ON    PARENT_SCOPE)
target_compile_definitions(chip-config PUBLIC -D TERMINAL_DEVICE_HTIF)

set(PLATFORM_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(chip-config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PLATFORM_INCLUDE})

target_link_libraries(chip-config PUBLIC rocketcore)
target_link_libraries(chip-config PUBLIC clint)
target_link_libraries(chip-config PUBLIC plic)
target_link_libraries(chip-config PUBLIC htif)
target_link_libraries(chip-config PUBLIC uart)
target_link_libraries(chip-config PUBLIC spi)
target_link_libraries(chip-config PUBLIC i2c)
target_link_libraries(chip-config PUBLIC gpio)
