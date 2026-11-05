add_dynamatic_library(DynamaticCfToHandshake
  CfToHandshake.cpp
  BaseCfg.cpp
  AnnotatedCfg.cpp
  ControlNetworkRouting.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticHandshake
  DynamaticBufferPlacement
  DynamaticTransforms
  MLIRIR
  MLIRPass
  )
