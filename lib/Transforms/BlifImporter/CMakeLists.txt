add_dynamatic_library(DynamaticBlifImporter
  BlifImporter.cpp
  BlifImporterSupport.cpp

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
