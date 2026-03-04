add_dynamatic_library(DynamaticBlifImporter
  BlifImporterSupport.cpp


  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticSupportUtils
  DynamaticHW
  DynamaticSynth
  MLIRIR
  LLVMSupport
)
