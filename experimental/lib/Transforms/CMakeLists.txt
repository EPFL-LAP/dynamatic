add_dynamatic_library(DynamaticExperimentalTransforms
  HandshakeFixArgNames.cpp
  HandshakePlaceBuffersCustom.cpp

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
