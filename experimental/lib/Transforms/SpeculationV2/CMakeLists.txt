include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/include/experimental/Transforms/SpeculationV2)

add_dynamatic_library(DynamaticSpeculationV2
  HandshakeSpeculationV2.cpp
  PostSpecV2.cpp
  PreSpecV2.cpp
  JSONImporter.cpp

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
  MaterializationUtil
  )

