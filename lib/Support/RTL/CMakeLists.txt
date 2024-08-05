add_dynamatic_library(DynamaticSupportRTL
  RTL.cpp
  RTLTypes.cpp

  LINK_LIBS PUBLIC
  DynamaticSupportUtils
  DynamaticSupportJSON
  DynamaticHW
  MLIRIR
  LLVMSupport
)
