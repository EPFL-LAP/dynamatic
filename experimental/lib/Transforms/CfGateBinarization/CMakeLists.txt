include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/include/experimental/Transforms/CfGateBinarization)

add_dynamatic_library(DynamaticCfGateBinarization
  CfGateBinarization.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRControlFlowDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
  DynamaticExperimentalSupport
  )

