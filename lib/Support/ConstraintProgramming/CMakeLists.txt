add_dynamatic_library(DynamaticConstraintProgramming
  ConstraintProgramming.cpp

  LINK_LIBS PUBLIC
  DynamaticSupportUtils
  DynamaticHW
  MLIRIR
  LLVMSupport
)
