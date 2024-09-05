add_dynamatic_library(DynamaticExperimentalConversion
  FtdCfToHandshake.cpp
  FtdMemoryInterface.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
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
  DynamaticExperimentalSupportBooleanLogic
  DynamaticSupportEspresso
  )

