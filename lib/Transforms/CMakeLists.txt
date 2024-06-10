add_dynamatic_library(DynamaticTransforms
  ArithReduceStrength.cpp
  BackAnnotate.cpp
  FlattenMemRefRowMajor.cpp
  ForceMemoryInterface.cpp
  FuncMaximizeSSA.cpp
  FuncSetArgNames.cpp
  HandshakeCanonicalize.cpp
  HandshakeConcretizeIndexType.cpp
  HandshakeHoistExtInstances.cpp
  HandshakeMaterialize.cpp
  HandshakeMinimizeCstWidth.cpp
  HandshakeMinimizeLSQUsage.cpp
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
  FuncSSAToGSA.cpp


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
