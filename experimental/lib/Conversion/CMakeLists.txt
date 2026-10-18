add_dynamatic_library(DynamaticExperimentalConversion
  FtdCfToHandshake.cpp
  OperandRoutingCfToHandshake.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticHandshake
  DynamaticBufferPlacement
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  DynamaticExperimentalSupportBooleanLogic
  DynamaticSupportEspresso
  DynamaticExperimentalSupport
  )

