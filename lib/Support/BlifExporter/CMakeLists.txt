add_dynamatic_library(DynamaticBlifExporter
  BlifExporterSupport.cpp

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticSupportUtils
  DynamaticHW
  DynamaticSynth
  MLIRIR
  LLVMSupport
)
