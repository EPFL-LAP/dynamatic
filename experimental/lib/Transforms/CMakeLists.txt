add_dynamatic_library(DynamaticExperimentalTransforms
  HandshakePlaceBuffersCustom.cpp
  HandshakeCombineSteeringLogic.cpp
  HandshakeStraightToQueue.cpp
  FtdProfiling.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRControlFlowDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
  DynamaticAnalysis
  DynamaticExperimentalAnalysis
)

add_subdirectory(Speculation)
add_subdirectory(ResourceSharing)
add_subdirectory(LSQSizing)
