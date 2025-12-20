add_dynamatic_library(DynamaticSupport
  Attribute.cpp
  Backedge.cpp
  CFG.cpp
  DOT.cpp
  Logging.cpp
  MILP.cpp
  System.cpp
  TimingModels.cpp
  DataflowGraph.cpp # TODO: Remove before release, this implementation is deprecated.

  LINK_LIBS PUBLIC
  DynamaticSupportDataflowGraph
  DynamaticSupportJSON
  DynamaticHandshake
  DynamaticHW
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
)

add_subdirectory(Espresso)
add_subdirectory(JSON)
add_subdirectory(RTL)
add_subdirectory(Utils)
add_subdirectory(DataflowGraph)