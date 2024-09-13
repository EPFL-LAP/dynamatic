add_dynamatic_library(DynamaticHandshakeToHW
  HandshakeToHW.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupportRTL
  DynamaticHandshake
  DynamaticHW
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
