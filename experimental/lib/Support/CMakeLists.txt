add_dynamatic_library(DynamaticExperimentalSupport
  CDGAnalysis.cpp
  StdProfiler.cpp
  HandshakeSimulator.cpp
  FtdSupport.cpp
  CutEnumeration.cpp
  BlifReader.cpp
  SubjectGraph.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect
)

add_subdirectory(BooleanLogic)
