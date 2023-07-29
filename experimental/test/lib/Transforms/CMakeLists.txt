add_dynamatic_library(DynamaticExperimentalTestTransforms
  TestCDGAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticSupport
  DynamaticExperimentalSupport
)
