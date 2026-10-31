add_dynamatic_library(DynamaticFormalPropertyAnnotation
  HandshakeAnnotateProperties.cpp
  Utils/FlowExpression.cpp
  Utils/Gaussian.cpp
  Utils/IndexChannelAnalysis.cpp
  Utils/IOG.cpp

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

