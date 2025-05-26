add_dynamatic_library(DynamaticExperimentalAnalysis
  GSAAnalysis.cpp

  DEPENDS
  DynamaticExperimentalAnalysisPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
  DynamaticExperimentalSupport
)

add_subdirectory(FormalPropertyAnnotation)
