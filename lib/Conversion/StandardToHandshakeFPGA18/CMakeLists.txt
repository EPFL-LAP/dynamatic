add_dynamatic_library(DynamaticStandardToHandshakeFPGA18
  StandardToHandshakeFPGA18.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  # ADDITIONAL_HEADER_DIRS
  # ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToHandshakeFPGA18

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTSupport
  CIRCTControlFlowLoopAnalysis
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
