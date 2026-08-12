add_dynamatic_library(DynamaticHandshakeToSynth
  HandshakeToSynth.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupportRTL
  DynamaticHandshake
  DynamaticSynth
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
