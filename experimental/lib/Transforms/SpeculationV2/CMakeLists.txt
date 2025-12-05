include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/include/experimental/Transforms/SpeculationV2)

add_dynamatic_library(DynamaticSpeculationV2
  HandshakeSpeculationV2.cpp
  PostSpecV2.cpp
  PreSpecV2.cpp
  JSONImporter.cpp
  SpecV2PostBuffering.cpp
  SpecV2CutCondDep.cpp
  SpecV2Gamma.cpp
  SpecV2GammaPostBuffering.cpp
  PreSpecV2Gamma.cpp
  SpecV2Lib.cpp
  SpecV1Adaptor.cpp
  SpecV1PostAdaptor.cpp
  HandshakeCopyBuffers.cpp
  HandshakeCutBBsByBuffers.cpp

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

