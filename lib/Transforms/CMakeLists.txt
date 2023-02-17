add_dynamatic_library(DynamaticTransforms
  HandshakeInferBasicBlocks.cpp
  HandshakeToDot.cpp
  PushConstants.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)
