add_subdirectory(experimental/xls)

# The XLS header include the tablegen'd files as follows:
# #include "xls/contrib/mlir/IR/xls_ops*"
# 
# The tablegen rules below generate the files in the following location:
# ${DYNAMATIC_BUILD_FOLDER}/experimental/xls/include/xls/contrib/mlir/IR/xls_ops*
#
# Since ${DYNAMATIC_BUILD_FOLDER}/experimental/xls/include is part of the include path,
# this works out.

# ===--- Dialect Ops ---===

set(LLVM_TARGET_DEFINITIONS ${DYNAMATIC_XLS_DIR}/xls/contrib/mlir/IR/xls_ops.td)
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops.h.inc           -gen-op-decls                             EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops.cc.inc          -gen-op-defs                              EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_attrs.h.inc     -gen-attrdef-decls --attrdefs-dialect=xls EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_attrs.cc.inc    -gen-attrdef-defs  --attrdefs-dialect=xls EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_enums.h.inc     -gen-enum-decls                           EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_enums.cc.inc    -gen-enum-defs                            EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_typedefs.h.inc  -gen-typedef-decls -typedefs-dialect=xls  EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/xls_ops_typedefs.cc.inc -gen-typedef-defs -typedefs-dialect=xls   EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
add_public_tablegen_target(MLIRXlsIncGen)
add_dependencies(mlir-headers MLIRXlsIncGen)
add_dependencies(dynamatic-headers MLIRXlsIncGen)

# ===--- Dialect Interfaces ---===

# Run tablegen for XLS interfaces:
set(LLVM_TARGET_DEFINITIONS ${DYNAMATIC_XLS_DIR}/xls/contrib/mlir/IR/interfaces.td)
mlir_tablegen(./xls/contrib/mlir/IR/interfaces.h.inc  -gen-op-interface-decls EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
mlir_tablegen(./xls/contrib/mlir/IR/interfaces.cc.inc -gen-op-interface-defs  EXTRA_INCLUDES ${DYNAMATIC_XLS_DIR})
add_public_tablegen_target(MLIRXlsInterfaceIncGen)
add_dependencies(mlir-generic-headers MLIRXlsInterfaceIncGen)
add_dependencies(dynamatic-headers MLIRXlsInterfaceIncGen)

