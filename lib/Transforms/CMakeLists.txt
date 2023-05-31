add_dynamatic_library(DynamaticTransforms
  ArithReduceArea.cpp
  AnalyzeMemoryAccesses.cpp
  FlattenMemRefRowMajor.cpp
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  NameMemoryOps.cpp
  PushConstants.cpp
  UtilsBitsUpdate.cpp
  OptimizeBits.cpp
  InitIndexType.cpp
  InitCstWidth.cpp
  ScfRotateForLoops.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)
