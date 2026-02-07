add_dynamatic_library(DynamaticLinearAlgebra
  Gaussian.cpp

  LINK_LIBS PUBLIC
  DynamaticSupportUtils
  DynamaticHW
  MLIRIR
  LLVMSupport
)
