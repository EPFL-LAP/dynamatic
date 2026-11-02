add_dynamatic_library(DynamaticBufferPlacement
  Utils/BufferingSupport.cpp
  Utils/BufferPlacementMILP.cpp
  Utils/CFDFC.cpp
  Utils/UnitMILPVars.cpp
  # TODO [amos]: Remove these two once the mix works
  FPGA20Buffers.cpp
  FPL22Buffers.cpp
  MixedBuffers.cpp
  FPGA24Buffers.cpp
  LatencyAndOccupancyBalancingSupport.cpp
  CostAwareBuffers.cpp
  HandshakePlaceBuffers.cpp
  HandshakeSetBufferingProperties.cpp
  MAPBUFBuffers.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  DynamaticConstraintProgramming

  )

if (GUROBI_FOUND)
  target_link_libraries(DynamaticBufferPlacement
    PUBLIC
    ${GUROBI_CXX_LIBRARY}
    ${GUROBI_LIBRARY}
  )
endif()

