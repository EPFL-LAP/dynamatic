add_dynamatic_library(DynamaticTestTransforms
  TestRTLSupport.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
)
