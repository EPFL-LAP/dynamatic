add_dynamatic_library(DynamaticBlifExporter
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
