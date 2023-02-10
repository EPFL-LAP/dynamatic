add_dynamatic_dialect_library(DynamaticHandshakeTransforms
  HandshakeToDot.cpp

  DEPENDS
  DynamaticHandshakeTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHandshake
  MLIRIR
  MLIRSCFDialect
  MLIRPass
  MLIRTransformUtils
  MLIRMemRefDialect
  MLIRAffineDialect
  )