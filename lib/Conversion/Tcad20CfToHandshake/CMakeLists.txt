add_dynamatic_library(DynamaticTcad20CfToHandshake
  Tcad20CfToHandshake.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticConversionUtils
  DynamaticSupport
  DynamaticHandshake
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
