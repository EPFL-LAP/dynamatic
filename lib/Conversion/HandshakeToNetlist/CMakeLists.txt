add_dynamatic_library(DynamaticHandshakeToNetlist
  HandshakeToNetlist.cpp

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
