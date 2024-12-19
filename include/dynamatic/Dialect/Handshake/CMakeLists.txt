add_dynamatic_dialect(Handshake handshake)
add_dynamatic_interface(HandshakeInterfaces)

set(LLVM_TARGET_DEFINITIONS Handshake.td)

mlir_tablegen(HandshakeEnums.h.inc -gen-enum-decls)
mlir_tablegen(HandshakeEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHandshakeEnumsIncGen)
add_dependencies(dynamatic-headers MLIRHandshakeEnumsIncGen)

mlir_tablegen(HandshakeAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=handshake)
mlir_tablegen(HandshakeAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=handshake)
add_public_tablegen_target(MLIRHandshakeAttributesIncGen)
add_dependencies(dynamatic-headers MLIRHandshakeAttributesIncGen)

# Generation of the type interfaces required an independent target
set(LLVM_TARGET_DEFINITIONS HandshakeTypeInterfaces.td)
mlir_tablegen(HandshakeTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(HandshakeTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRHandshakeTypeInterfacesIncGen)
add_dependencies(dynamatic-headers MLIRHandshakeTypeInterfacesIncGen)
