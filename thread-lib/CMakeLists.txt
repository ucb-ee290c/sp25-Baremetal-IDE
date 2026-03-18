set(THREAD_LIB_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/hthread.c
)
add_library(threadlib STATIC ${THREAD_LIB_SRC})
target_include_directories(threadlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(threadlib PUBLIC clint rocketcore chip-config)