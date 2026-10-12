add_dynamatic_library(DynamaticHandshakeToHW
  HandshakeToHW.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticAnalysis
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
