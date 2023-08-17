add_dynamatic_library(DynamaticTransforms
  ArithReduceStrength.cpp
  AnalyzeMemoryAccesses.cpp
  FlattenMemRefRowMajor.cpp
  HandshakeConcretizeIndexType.cpp
  HandshakeMinimizeCstWidth.cpp
  HandshakeOptimizeBitwidths.cpp
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
  DynamaticSupport
  DynamaticAnalysis
)

add_subdirectory(BufferPlacement)
