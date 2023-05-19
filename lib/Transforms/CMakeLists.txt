add_dynamatic_library(DynamaticTransforms
  # ArithOptimizeArea.cpp
  AnalyzeMemoryAccesses.cpp
  FlattenMemRefRowMajor.cpp
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  HandshakePlaceBuffers.cpp
  NameMemoryOps.cpp
  PushConstants.cpp
  UtilsForMILPSolver.cpp
  UtilsForPlaceBuffers.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)

target_include_directories(DynamaticTransforms 
  PUBLIC
  "/opt/gurobi1001/linux64/include/"
)
