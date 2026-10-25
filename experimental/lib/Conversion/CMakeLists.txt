add_dynamatic_library(DynamaticExperimentalConversion
  FtdCfToHandshake.cpp
  OperandRoutingCfToHandshake.cpp
  OperandRouting2/OperandRouting2CfToHandshake.cpp
  OperandRouting2/BaseCfg.cpp
  OperandRouting2/AnnotatedCfg.cpp
  OperandRouting2/ControlNetworkRouting.cpp

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

