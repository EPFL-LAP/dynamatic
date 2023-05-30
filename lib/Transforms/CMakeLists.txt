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

include_directories(${GUROBI_INCLUDE_DIRS})

target_link_libraries(DynamaticTransforms PUBLIC 
  ${GUROBI_CXX_LIBRARY}
)

target_link_libraries(DynamaticTransforms PUBLIC 
  ${GUROBI_LIBRARY}
)



