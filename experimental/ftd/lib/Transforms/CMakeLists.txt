add_dynamatic_library(DynamaticFTDTransforms
  HandshakeCombineSteeringLogic.cpp

  DEPENDS
  DynamaticFTDTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
)