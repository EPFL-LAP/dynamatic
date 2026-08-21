add_dynamatic_library(DynamaticTransforms
  AggregateMemory.cpp
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
  HandshakeMarkFPUImpl.cpp
  HWMarkLIMemoryInterface.cpp
  MarkMemoryDependencies.cpp
  MarkMemoryInterfaces.cpp
  OperationNames.cpp
  PushConstants.cpp
  RemovePolygeistAttributes.cpp
  ScfRotateForLoops.cpp
  ScfSimpleIfToSelect.cpp
  DropUnlistedFunctions.cpp
  SystolicUnitGeneration.cpp
  SystolicUnitReuseOptimization.cpp

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

target_link_libraries(DynamaticTransforms
  PUBLIC ${Z3_LIB}
)
