add_dynamatic_library(DynamaticLSQSizing
  HandshakeSizeLSQs.cpp
  LSQSizingSupport.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  DynamaticBufferPlacement
  )
