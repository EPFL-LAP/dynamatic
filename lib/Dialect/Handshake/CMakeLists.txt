add_dynamatic_dialect_library(DynamaticHandshake
  HandshakeCanonicalize.cpp
  HandshakeOps.cpp
  HandshakeDialect.cpp
  HandshakeInterfaces.cpp
  HandshakeTypes.cpp
  MemoryInterfaces.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRArithDialect
  
  DEPENDS
  MLIRHandshakeInterfacesIncGen
  MLIRHandshakeAttributesIncGen
  MLIRHandshakeEnumsIncGen
  )
