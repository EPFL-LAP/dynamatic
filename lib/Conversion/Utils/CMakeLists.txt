add_dynamatic_library(DynamaticConversionUtils
  CfToHandshakeTranslator.cpp
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
  DynamaticSupportBooleanLogic
  DynamaticSupportEspresso
  DynamaticFtd
  DynamaticAnalysis
  )
