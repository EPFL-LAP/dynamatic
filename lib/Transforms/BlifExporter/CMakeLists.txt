add_dynamatic_library(DynamaticBlifExporter
  BlifExporter.cpp
  BlifExporterSupport.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticSupportUtils
  DynamaticHW
  DynamaticSynth
  MLIRIR
  LLVMSupport
)
