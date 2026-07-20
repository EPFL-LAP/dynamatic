include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/include/experimental/Transforms/EagerlyElastic)
add_dynamatic_library(DynamaticEagerlyElastic
  EagerlyElasticAD.cpp
  EagerlyElasticAll.cpp
  EagerlyElasticLib.cpp

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
