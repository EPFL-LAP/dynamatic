add_dynamatic_library(DynamaticTransforms
  # ArithOptimizeArea.cpp
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

  BufferPlacement/HandshakePlaceBuffers.cpp
  BufferPlacement/ExtractMG.cpp
  BufferPlacement/OptimizeMILP.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)

target_link_libraries(DynamaticTransforms PUBLIC 
  ${GUROBI_CXX_LIBRARY}
)

target_link_libraries(DynamaticTransforms PUBLIC 
  ${GUROBI_LIBRARY}
)