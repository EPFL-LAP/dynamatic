add_dynamatic_library(DynamaticLinearAlgebra
  Gaussian.cpp

  LINK_LIBS PUBLIC
  MLIRIR

  LINK_COMPONENTS
  Support
)
