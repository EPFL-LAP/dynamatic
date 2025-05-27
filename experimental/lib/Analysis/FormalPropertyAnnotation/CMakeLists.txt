add_dynamatic_library(DynamaticFormalPropertyAnnotation
  HandshakeAnnotateProperties.cpp

  DEPENDS
  DynamaticExperimentalAnalysisPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  )

