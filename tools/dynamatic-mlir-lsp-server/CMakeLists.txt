add_llvm_tool(dynamatic-mlir-lsp-server
  dynamatic-mlir-lsp-server.cpp
)

llvm_update_compile_flags(dynamatic-mlir-lsp-server)
target_link_libraries(dynamatic-mlir-lsp-server
  PRIVATE
  DynamaticHandshake
  DynamaticHW
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLspServerLib
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
)
