include_directories(${DYNAMATIC_SOURCE_DIR}/include/dynamatic/Transforms/Speculation)

add_dynamatic_library(DynamaticSpeculation
  SpeculationPlacement.cpp
  HandshakeSpeculation.cpp
  HandshakeSpecPostBuffer.cpp
  PlacementFinder.cpp

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
  )

