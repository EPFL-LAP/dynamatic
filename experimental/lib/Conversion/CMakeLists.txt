add_dynamatic_library(DynamaticExperimentalConversion
  StandardToHandshakeFPL22.cpp

  DEPENDS
  DynamaticExperimentalConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  DynamaticSupport
)