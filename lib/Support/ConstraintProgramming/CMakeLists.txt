add_dynamatic_library(DynamaticConstraintProgramming
  ConstraintProgramming.cpp

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticSupportUtils
  DynamaticHW
  MLIRIR
  LLVMSupport
)
