add_dynamatic_library(DynamaticExperimentalTransforms
  HandshakePlaceBuffersCustom.cpp
  HandshakeRigidification.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
  DynamaticAnalysis
)

add_subdirectory(Speculation)
add_subdirectory(ResourceSharing)
add_subdirectory(LSQSizing)
