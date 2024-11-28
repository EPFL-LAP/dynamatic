add_dynamatic_library(DynamaticExperimentalSupport
  StdProfiler.cpp
  HandshakeSimulator.cpp
  FtdSupport.cpp
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
