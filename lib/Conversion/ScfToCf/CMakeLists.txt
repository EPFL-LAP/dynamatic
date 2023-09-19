add_dynamatic_library(DynamaticLowerScfToCf
  ScfToCf.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  MLIRSCFToControlFlow
  DynamaticSupport
  )
