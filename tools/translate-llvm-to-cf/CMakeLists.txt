set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(translate-llvm-to-cf
  translate-llvm-to-cf.cpp
)

llvm_update_compile_flags(translate-llvm-to-cf)
target_link_libraries(translate-llvm-to-cf
  PRIVATE

  MLIRIR
  MLIRAffineTransforms
  MLIRControlFlowDialect
  MLIRArithTransforms
  MLIRSCFTransforms
  MLIRMemRefTransforms
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect

  LLVMCore
  LLVMSupport
  LLVMAnalysis
)


if(DYNAMATIC_ENABLE_XLS)
  target_link_libraries(dynamatic-opt
    PRIVATE
    XlsDialect
    DynamaticHandshakeToXls
  )
endif()
