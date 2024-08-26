add_dynamatic_library(DynamaticExperimentalTestTransforms
  TestCDGAnalysis.cpp
  TestHandshakeSimulator.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
  DynamaticHandshake
  DynamaticExperimentalSupport
)
