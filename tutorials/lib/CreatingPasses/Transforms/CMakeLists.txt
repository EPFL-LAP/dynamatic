add_dynamatic_library(DynamaticTutorialsCreatingPasses
  SimplifyMergeLike.cpp

  DEPENDS
  DynamaticTutorialsCreatingPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)
