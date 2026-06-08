add_dynamatic_library(DynamaticBufferPlacement
  Utils/BufferingSupport.cpp
  Utils/BufferPlacementMILP.cpp
  Utils/CFDFC.cpp
  Utils/UnitMILPVars.cpp
  FPGA20Buffers.cpp
  FPL22Buffers.cpp
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

