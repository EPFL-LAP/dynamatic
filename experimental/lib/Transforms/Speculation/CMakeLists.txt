include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/include/experimental/Transforms/Speculation)

add_dynamatic_library(DynamaticSpeculation
  SpeculationPlacement.cpp
  HandshakeSpeculation.cpp
  PlacementFinder.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  )

