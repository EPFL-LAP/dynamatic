add_dynamatic_library(DynamaticOperandRouting2CfToHandshake
  OperandRouting2CfToHandshake.cpp
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
