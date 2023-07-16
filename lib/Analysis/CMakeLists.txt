add_dynamatic_library(DynamaticAnalysis
  ConstantAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
)
