add_dynamatic_library(DynamaticFtdCfToHandshake
  FtdCfToHandshake.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticConversionUtils
  DynamaticAnalysis
  MLIRIR
  MLIRPass
  )
