add_dynamatic_library(DynamaticSupportDataflowGraph
  DataflowGraphBase.cpp
  ReconvergentPathFinder.cpp

  LINK_LIBS PUBLIC
  DynamaticHandshake
  MLIRIR
  MLIRSupport
)