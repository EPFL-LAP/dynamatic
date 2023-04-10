add_dynamatic_library(DynamaticTransforms
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  PushConstants.cpp
  BitsOptimize.cpp
  InitIndexType.cpp
  InitCstWidth.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)
