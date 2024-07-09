add_dynamatic_library(DynamaticTutorialsCreatingPasses
  GreedySimplifyMergeLike.cpp
  SimplifyMergeLike.cpp

  DEPENDS
  DynamaticTutorialsCreatingPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)
