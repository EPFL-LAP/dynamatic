add_dynamatic_library(DynamaticExperimentalAnalysis
  GSAAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
  DynamaticExperimentalSupport
)

add_subdirectory(FormalPropertyAnnotation)
