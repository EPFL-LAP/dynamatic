add_dynamatic_library(DynamaticTransforms
  ArithReduceArea.cpp
  AnalyzeMemoryAccesses.cpp
  FlattenMemRefRowMajor.cpp
  HandshakeConcretizeIndexType.cpp
  HandshakeMinimizeCstWidth.cpp
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  NameMemoryOps.cpp
  PushConstants.cpp
  UtilsBitsUpdate.cpp
  OptimizeBits.cpp
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
