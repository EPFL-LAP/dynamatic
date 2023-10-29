add_dynamatic_library(DynamaticAnalysis
  ConstantAnalysis.cpp
  NameAnalysis.cpp
  NumericAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticSupport
)
