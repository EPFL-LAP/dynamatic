add_dynamatic_library(DynamaticExperimentalSupport
  BlifReader.cpp
  CDGAnalysis.cpp
  StdProfiler.cpp
  FtdSupport.cpp
  HandshakeSimulator.cpp
  FtdImplementation.cpp
  CFGAnnotation.cpp

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
