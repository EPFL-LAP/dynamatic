add_dynamatic_library(DynamaticLSQSizing
  HandshakeSizeLSQ.cpp
  Support/LSQSizingSupport.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticAnalysis
  DynamaticBufferPlacement
  )
