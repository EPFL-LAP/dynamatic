add_dynamatic_library(DynamaticTransforms
  AnalyzeMemoryAccesses.cpp
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  NameMemoryOps.cpp
  PushConstants.cpp
  BitsOptimize.cpp
  InitIndexType.cpp
  InitCstWidth.cpp
  UtilsBitsUpdate.cpp
  ForwardUpdate.cpp
  BackwardUpdate.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)
