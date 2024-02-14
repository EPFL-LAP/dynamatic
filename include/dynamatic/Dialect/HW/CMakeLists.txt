add_dynamatic_dialect(HW hw)

set(LLVM_TARGET_DEFINITIONS HW.td)

mlir_tablegen(HWAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(HWAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRHWAttrIncGen)
add_dependencies(dynamatic-headers MLIRHWAttrIncGen)

mlir_tablegen(HWEnums.h.inc -gen-enum-decls)
mlir_tablegen(HWEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHWEnumsIncGen)
add_dependencies(dynamatic-headers MLIRHWEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS HWOpInterfaces.td)
mlir_tablegen(HWOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HWOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(DynamaticHWOpInterfacesIncGen)
add_dependencies(dynamatic-headers DynamaticHWOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS HWTypeInterfaces.td)
mlir_tablegen(HWTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(HWTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(DynamaticHWTypeInterfacesIncGen)
add_dependencies(dynamatic-headers DynamaticHWTypeInterfacesIncGen)
