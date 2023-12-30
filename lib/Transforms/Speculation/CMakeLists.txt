add_dynamatic_library(DynamaticSpeculation
  SpeculationFir.cpp
  SpeculationPlacement.cpp
  HandshakeSpeculation.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  CIRCTHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  )

