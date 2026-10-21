add_dynamatic_library(DynamaticDuplication
  PredictedConstantDuplication.cpp
  LoopPrediction.cpp

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
  