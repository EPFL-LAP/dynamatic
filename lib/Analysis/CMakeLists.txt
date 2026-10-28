add_dynamatic_library(DynamaticAnalysis
  ControlNetworkAnalysis.cpp
  IndexChannelAnalysis.cpp
  NameAnalysis.cpp
  NumericAnalysis.cpp
  ControlDependenceAnalysis.cpp
  GSAAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
  DynamaticFtdSupport
)

add_subdirectory(FormalPropertyAnnotation)
