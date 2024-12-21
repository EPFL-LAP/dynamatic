set(LLVM_TARGET_DEFINITIONS HandshakeCanonicalization.td)
mlir_tablegen(HandshakeCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(DynamaticHandshakeCanonicalizationIncGen)

add_dynamatic_dialect_library(DynamaticHandshake
  HandshakeAttributes.cpp
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
  MLIRHandshakeTypeInterfacesIncGen
  MLIRHandshakeEnumsIncGen
  )
