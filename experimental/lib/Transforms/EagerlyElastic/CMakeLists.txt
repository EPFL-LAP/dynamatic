add_dynamatic_library(DynamaticEagerlyElastic
  EagerlyElasticAD.cpp
  EagerlyElasticPatternRewriting.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  DynamaticBufferPlacement
  )
