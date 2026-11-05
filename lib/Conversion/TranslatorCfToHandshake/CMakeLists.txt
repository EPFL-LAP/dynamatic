add_dynamatic_library(DynamaticTranslatorCfToHandshake
  CfToHandshakeTranslator.cpp
  FtdCfToHandshakeTranslator.cpp
  Tcad20CfToHandshake.cpp
  FtdCfToHandshake.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticHandshake
  DynamaticTransforms
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
  DynamaticExperimentalAnalysis
  )
