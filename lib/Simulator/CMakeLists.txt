add_dynamatic_library(DynamaticSimulator
  Simulator.cpp

  LINK_LIBS PUBLIC
  DynamaticHandshake
  MLIRIR
  MLIRSupport
)
