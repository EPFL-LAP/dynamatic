if (NOT GUROBI_FOUND)
  add_compile_definitions(DYNAMATIC_GUROBI_NOT_INSTALLED)
endif()

add_dynamatic_library(DynamaticBufferPlacement
  BufferingProperties.cpp
  ExtractCFDFC.cpp
  HandshakePlaceBuffers.cpp
  OptimizeMILP.cpp
  ParseCircuitJson.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
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
