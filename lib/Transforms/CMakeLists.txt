add_dynamatic_library(DynamaticTransforms
  ArithReduceStrength.cpp
  AnalyzeMemoryAccesses.cpp
  BackAnnotate.cpp
  FlattenMemRefRowMajor.cpp
  HandshakeCanonicalize.cpp
  HandshakeConcretizeIndexType.cpp
  HandshakeMinimizeCstWidth.cpp
  HandshakeOptimizeBitwidths.cpp
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  NameMemoryOps.cpp
  OperationNames.cpp
  PushConstants.cpp
  ScfRotateForLoops.cpp
  ScfSimpleIfToSelect.cpp

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
