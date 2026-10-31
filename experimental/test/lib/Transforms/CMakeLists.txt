add_dynamatic_library(DynamaticExperimentalTestTransforms
  TestHandshakeSimulator.cpp
  TestPrintFormalProperties.cpp

  DEPENDS
  DynamaticExperimentalTestTransformsPassIncGen

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
