add_dynamatic_library(DynamaticAnalysis
  NameAnalysis.cpp
  NumericAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
)
