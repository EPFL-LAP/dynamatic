add_dynamatic_library(DynamaticTransforms
  ArithReduceStrength.cpp
  BackAnnotate.cpp
  FlattenMemRefRowMajor.cpp
  ForceMemoryInterface.cpp
  HandshakeCanonicalize.cpp
  HandshakeConcretizeIndexType.cpp
  HandshakeMinimizeCstWidth.cpp
  HandshakeOptimizeBitwidths.cpp
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  MarkMemoryDependencies.cpp
  MarkMemoryInterfaces.cpp
  OperationNames.cpp
  PushConstants.cpp
  RemovePolygeistAttributes.cpp
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
