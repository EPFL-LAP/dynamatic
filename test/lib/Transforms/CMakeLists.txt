add_dynamatic_library(DynamaticTestTransforms
  TestRTLSupport.cpp

  DEPENDS
  DynamaticTestTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
)
