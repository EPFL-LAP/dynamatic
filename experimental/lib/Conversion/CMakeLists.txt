add_dynamatic_library(DynamaticExperimentalStandardToHandshakeFPL22
  StandardToHandshakeFPL22.cpp

  DEPENDS
  DynamaticExperimentalConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  DynamaticSupport
)
