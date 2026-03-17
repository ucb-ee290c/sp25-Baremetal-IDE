
#################################
# Build
#################################

add_executable(read-test
  src/main.c
)

target_include_directories(read-test PUBLIC include)
target_include_directories(read-test PUBLIC ${CMAKE_SOURCE_DIR}/platform/c2c25/include)

target_link_libraries(read-test PRIVATE 
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

target_link_libraries(read-test PRIVATE libbmark)