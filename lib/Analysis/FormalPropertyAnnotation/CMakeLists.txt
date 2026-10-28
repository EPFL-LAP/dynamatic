add_dynamatic_library(DynamaticFormalPropertyAnnotation
  HandshakeAnnotateProperties.cpp

  DEPENDS
  DynamaticAnalysisPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticFormalVerification
  )

