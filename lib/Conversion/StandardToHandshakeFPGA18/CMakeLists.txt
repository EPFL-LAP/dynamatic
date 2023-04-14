add_dynamatic_library(DynamaticStandardToHandshakeFPGA18
  StandardToHandshakeFPGA18.cpp

  DEPENDS
  DynamaticConversionPassIncGen

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
  )
