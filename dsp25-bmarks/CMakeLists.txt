add_subdirectory(1dconv-bmarks)

if(BUILD_MFCC_LIB)
  add_subdirectory(mfcc-bmarks)
  add_subdirectory(mfcc-bmarks-mc)
endif()
