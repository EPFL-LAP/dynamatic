set_property(GLOBAL APPEND PROPERTY DYNAMATIC_DIALECT_LIBS XlsDialect)

add_mlir_library(XlsDialect
  ${DYNAMATIC_XLS_DIR}/xls/contrib/mlir/IR/xls_ops.cc
  ${DYNAMATIC_XLS_DIR}/xls/contrib/mlir/IR/assembly_format.cc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSupport
  MLIRAffineDialect

  DEPENDS
  MLIRXlsIncGen
  MLIRXlsInterfaceIncGen
  dynamatic-headers
)

# XLS sources feature include paths relative to `llvm-project`, not relative
# to `llvm-project/llvm/include` and `llvm-project/mlir/include`.
# I am so sorry...
foreach(LLVM_INCLUDE_DIR IN LISTS LLVM_INCLUDE_DIRS)
  target_include_directories(XlsDialect PUBLIC ${LLVM_INCLUDE_DIR}/../..)
endforeach()
