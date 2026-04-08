add_subdirectory(dma-bmarks)
add_subdirectory(simple-conv-bmark)

if(BUILD_MFCC_LIB)
  add_subdirectory(mfcc-bmarks)
  if(TARGET threadlib)
    add_subdirectory(mfcc-bmarks-mc-threadlib)
  endif()
endif()
