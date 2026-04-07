
#################################
# Build
#################################

add_executable(membw-bmark
  src/main.c
  src/mtwister.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

target_include_directories(membw-bmark PUBLIC include)
target_include_directories(membw-bmark PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(membw-bmark PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

target_link_libraries(membw-bmark PRIVATE chip-config)
target_link_libraries(membw-bmark PRIVATE rocketcore)

if (TARGET threadlib)
  target_include_directories(membw-bmark BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/thread-lib)
  target_link_libraries(membw-bmark PRIVATE threadlib)
  target_compile_definitions(membw-bmark PRIVATE BW_USE_THREADLIB=1)
else()
  message(STATUS "membw-bmark: THREAD_LIB is OFF, disabling thread-lib multicore backend")
  target_compile_definitions(membw-bmark PRIVATE BW_USE_THREADLIB=0)
endif()

add_executable(membw-bmark-uc
  src/membw-bmark-uc.c
  src/mtwister.c
)

target_include_directories(membw-bmark-uc PUBLIC include)
target_include_directories(membw-bmark-uc PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(membw-bmark-uc PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

target_link_libraries(membw-bmark-uc PRIVATE libbmark)
target_link_libraries(membw-bmark-uc PRIVATE chip-config)
target_link_libraries(membw-bmark-uc PRIVATE rocketcore)
