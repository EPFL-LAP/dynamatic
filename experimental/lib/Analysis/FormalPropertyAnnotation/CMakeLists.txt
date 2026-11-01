add_dynamatic_library(DynamaticFormalPropertyAnnotation
  HandshakeAnnotateProperties.cpp
  Utils/Elimination.cpp
  Utils/FlowExpression.cpp
  Utils/IndexChannelAnalysis.cpp
  Utils/IOG.cpp
  Utils/TraversalUtils.cpp

  DEPENDS
  DynamaticExperimentalAnalysisPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRPresburger
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  )

