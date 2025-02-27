set_property(GLOBAL APPEND PROPERTY DYNAMATIC_DIALECT_LIBS XlsDialect)

add_mlir_library(XlsDialect
xls_ops.cc
assembly_format.cc

LINK_LIBS PUBLIC
MLIRIR
MLIRFuncDialect
MLIRArithDialect
MLIRSupport
MLIRAffineDialect

DEPENDS
MLIRXlsIncGen
MLIRXlsInterfaceIncGen
MLIRXlsAllHeaders
dynamatic-headers
)

# XLS sources feature include paths relative to `llvm-project`, not relative
# to `llvm-project/llvm/include` and `llvm-project/mlir/include`.
# I am so sorry...
foreach(LLVM_INCLUDE_DIR IN LISTS LLVM_INCLUDE_DIRS)
  target_include_directories(XlsDialect PUBLIC ${LLVM_INCLUDE_DIR}/../..)
endforeach()

target_link_libraries(XlsDialect PUBLIC MLIRXlsAllHeaders)
