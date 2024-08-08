add_dynamatic_library(DynamaticAnalysis
  ConstantAnalysis.cpp
  NameAnalysis.cpp
  NumericAnalysis.cpp
  ControlDependenceAnalysis.cpp
  GsaAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
)
