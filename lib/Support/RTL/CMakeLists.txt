add_dynamatic_library(DynamaticSupportRTL
  RTL.cpp

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticHW
  MLIRIR
  LLVMSupport
)
