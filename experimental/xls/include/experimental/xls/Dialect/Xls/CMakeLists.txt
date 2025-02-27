# ===--- Dialect Ops ---===

set(DYNAMATIC_XLS_DIALECT_BIN_DIR ${DYNAMATIC_BINARY_DIR}/experimental/xls/include/experimental/xls/Dialect/Xls/)

set(LLVM_TARGET_DEFINITIONS xls/contrib/mlir/IR/xls_ops.td)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops.h.inc           -gen-op-decls EXTRA_INCLUDES .)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops.cc.inc          -gen-op-defs EXTRA_INCLUDES .)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_attrs.h.inc     -gen-attrdef-decls --attrdefs-dialect=xls EXTRA_INCLUDES .)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_attrs.cc.inc    -gen-attrdef-defs  --attrdefs-dialect=xls EXTRA_INCLUDES .)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_enums.h.inc     -gen-enum-decls EXTRA_INCLUDES .)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_enums.cc.inc    -gen-enum-defs EXTRA_INCLUDES .)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_typedefs.h.inc  -gen-typedef-decls -typedefs-dialect=xls EXTRA_INCLUDES .)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_typedefs.cc.inc -gen-typedef-defs -typedefs-dialect=xls EXTRA_INCLUDES .)
add_public_tablegen_target(MLIRXlsIncGen)
add_dependencies(mlir-headers MLIRXlsIncGen)
add_dependencies(dynamatic-headers MLIRXlsIncGen)

# ===--- Dialect Interfaces ---===

# Run tablegen for XLS interfaces:
set(LLVM_TARGET_DEFINITIONS xls/contrib/mlir/IR/interfaces.td)
mlir_tablegen(./xls/contrib/mlir/IR/interfaces.h.inc  -gen-op-interface-decls)
mlir_tablegen(./xls/contrib/mlir/IR/interfaces.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRXlsInterfaceIncGen)
add_dependencies(mlir-generic-headers MLIRXlsInterfaceIncGen)
add_dependencies(dynamatic-headers MLIRXlsInterfaceIncGen)

# ===--- Dialect Headers Bundle ---===

# Provide library that bundles all headers (provided + generated) for this
# dialect, to keep the integration path mess contained to this file:
add_library(MLIRXlsAllHeaders INTERFACE)
target_include_directories(MLIRXlsAllHeaders INTERFACE .)
target_include_directories(MLIRXlsAllHeaders INTERFACE ${DYNAMATIC_XLS_DIALECT_BIN_DIR})
add_dependencies(MLIRXlsAllHeaders MLIRXlsIncGen)
add_dependencies(MLIRXlsAllHeaders MLIRXlsInterfaceIncGen)
