add_dynamatic_library(VisualizerSupport
  VisualizerSupport.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  DynamaticHandshake
)
add_subdirectory(log2csv)
add_subdirectory(cex2csv)
