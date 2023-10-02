if (NOT GUROBI_FOUND)
  add_compile_definitions(DYNAMATIC_GUROBI_NOT_INSTALLED)
endif()

add_dynamatic_library(DynamaticBufferPlacement
  BufferingProperties.cpp
  BufferPlacementMILP.cpp
  CFDFC.cpp
  HandshakePlaceBuffers.cpp
  HandshakeSetBufferingProperties.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  CIRCTHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  )

if (GUROBI_FOUND)
  target_link_libraries(DynamaticBufferPlacement
    PUBLIC
    ${GUROBI_CXX_LIBRARY}
    ${GUROBI_LIBRARY}
  )
endif()
