add_dynamatic_library(DynamaticTestTransforms
  TestRTLSupport.cpp
  TestHandshakeSimulator.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
  DynamaticHandshake
  DynamaticSimulation
)
