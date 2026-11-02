add_dynamatic_library(DynamaticConversionUtils
  FtdCfToHandshakeTranslator.cpp

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticHandshake
  DynamaticTransforms
  MLIRIR
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  DynamaticExperimentalSupportBooleanLogic
  DynamaticSupportEspresso
  DynamaticExperimentalSupport
  DynamaticExperimentalAnalysis
  )
