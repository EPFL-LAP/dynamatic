add_dynamatic_library(DynamaticExperimentalTransforms
  HandshakeFixArgNames.cpp
  HandshakeRemoveMemories.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
  DynamaticAnalysis
)
