add_dynamatic_dialect_library(DynamaticHandshake
  HandshakeOps.cpp
  HandshakeDialect.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRArithDialect
  
  DEPENDS
  MLIRHandshakeInterfacesIncGen
  MLIRHandshakeAttributesIncGen
  MLIRHandshakeEnumsIncGen
  )
