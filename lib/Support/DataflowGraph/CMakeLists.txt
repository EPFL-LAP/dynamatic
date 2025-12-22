add_dynamatic_library(DynamaticSupportDataflowGraph
  DataflowGraphBase.cpp
  ReconvergentPathFinder.cpp
  SynchronizingCyclesFinder.cpp

  LINK_LIBS PUBLIC
  DynamaticHandshake
  MLIRIR
  MLIRSupport
)