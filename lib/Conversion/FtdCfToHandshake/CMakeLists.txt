add_dynamatic_library(DynamaticFtdCfToHandshake
  FtdCfToHandshake.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticConversionUtils
  DynamaticExperimentalAnalysis
  MLIRIR
  MLIRPass
  )
