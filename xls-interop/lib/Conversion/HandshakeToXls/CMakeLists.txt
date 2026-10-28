add_dynamatic_library(DynamaticHandshakeToXls
  HandshakeToXls.cpp

  DEPENDS
  DynamaticXlsConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupportRTL
  DynamaticHandshake
  XlsDialect
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
)
