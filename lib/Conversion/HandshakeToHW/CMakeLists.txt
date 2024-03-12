add_dynamatic_library(DynamaticHandshakeToHW
  HandshakeToHW.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
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
