add_dynamatic_library(DynamaticExperimentalTransforms
  FuncSetArgNames.cpp
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
  libclang
)

add_subdirectory(Speculation)
