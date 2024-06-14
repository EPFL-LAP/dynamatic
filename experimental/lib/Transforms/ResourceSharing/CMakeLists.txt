add_dynamatic_library(DynamaticExperimentalResourceSharing
  Crush.cpp
  SharingSupport.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  )
