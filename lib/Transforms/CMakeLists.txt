add_dynamatic_library(DynamaticTransforms
  ArithReduceStrength.cpp
  BackAnnotate.cpp
  FlattenMemRefRowMajor.cpp
  ForceMemoryInterface.cpp
  FuncMaximizeSSA.cpp
  FuncSetArgNames.cpp
  HandshakeAnalyzeLSQUsage.cpp
  HandshakeCanonicalize.cpp
  HandshakeHoistExtInstances.cpp
  HandshakeMaterialize.cpp
  HandshakeMinimizeCstWidth.cpp
  HandshakeOptimizeBitwidths.cpp
  HandshakeInferBasicBlocks.cpp
  HandshakeReplaceMemoryInterfaces.cpp
  HandshakeReshapeChannels.cpp
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
  libclang
)

add_subdirectory(BufferPlacement)
