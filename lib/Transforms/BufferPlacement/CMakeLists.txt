add_dynamatic_library(DynamaticBufferPlacement
  BufferingSupport.cpp
  BufferPlacementMILP.cpp
  CFDFC.cpp
  FPGA20Buffers.cpp
  FPL22Buffers.cpp
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

if (CBC_FOUND AND CLPCORE_FOUND AND OSI_FOUND)
target_link_libraries(DynamaticBufferPlacement
  PUBLIC
  ${CBC_LIBRARIES}
  ${CLPCORE_LIBRARIES}
  ${OSI_LIBRARIES}
)
endif()


