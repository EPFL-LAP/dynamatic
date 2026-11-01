add_dynamatic_library(DynamaticAnalysis
  ControlNetworkAnalysis.cpp
  NameAnalysis.cpp
  NumericAnalysis.cpp
  ControlDependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
)
