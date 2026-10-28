add_dynamatic_library(DynamaticSupport
  Attribute.cpp
  Backedge.cpp
  BLIFFileManager.cpp
  CFG.cpp
  DOT.cpp
  MILP.cpp
  System.cpp
  TimingModels.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  DynamaticSupportJSON
  DynamaticHandshake
  DynamaticLinearAlgebra
  DynamaticHW
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)

add_subdirectory(Espresso)
add_subdirectory(JSON)
add_subdirectory(RTL)
add_subdirectory(Utils)
add_subdirectory(ConstraintProgramming)
add_subdirectory(BlifImporter)
add_subdirectory(BlifExporter)
add_subdirectory(LinearAlgebra)

# The libraries below share this directory with DynamaticSupport
# (PARTIAL_SOURCES_INTENDED) but are separate from it: most of them use analyses
# or transforms, which depend on DynamaticSupport

# Technology mapping of BLIF netlists onto LUTs (used by MapBuf)
add_dynamatic_library(DynamaticTechnologyMapping
  BlifReader.cpp
  CutlessMapping.cpp
  SubjectGraph.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  DynamaticAnalysis
  DynamaticHandshake
  DynamaticSupport
)

# Formal properties and SMV testbenches for formal verification
add_dynamatic_library(DynamaticFormalVerification
  CreateSmvFormalTestbench.cpp
  FlowExpression.cpp
  FormalProperty.cpp
  IOG.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  DynamaticAnalysis
  DynamaticHandshake
  DynamaticSupport
)

# Simulators of std-level and Handshake-level IR
add_dynamatic_library(DynamaticSimulation
  HandshakeSimulator.cpp
  StdProfiler.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRParser
  MLIRSupport
  DynamaticHandshake
  DynamaticSupport
)

# Fast token delivery utilities that GSAAnalysis uses, separate from
# DynamaticFtd since DynamaticFtd uses GSAAnalysis
add_dynamatic_library(DynamaticFtdSupport
  FtdSupport.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  DynamaticHandshake
  DynamaticSupport
  DynamaticSupportBooleanLogic
)

# Fast token delivery
add_dynamatic_library(DynamaticFtd
  CFGAnnotation.cpp
  FtdImplementation.cpp
  FtdSuppression.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSupport
  MLIRTransforms
  DynamaticAnalysis
  DynamaticFtdSupport
  DynamaticHandshake
  DynamaticSupport
  DynamaticSupportBooleanLogic
)

add_subdirectory(BooleanLogic)
