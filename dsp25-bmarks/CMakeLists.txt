add_subdirectory(1dconv-bmarks)

if(BUILD_MFCC_LIB)
  add_subdirectory(mfcc-bmarks)
  if(TARGET threadlib)
    add_subdirectory(mfcc-bmarks-mc-threadlib)
  endif()
endif()
