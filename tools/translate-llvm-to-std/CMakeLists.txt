set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(translate-llvm-to-std
  main.cpp
  TranslateLLVMToStd.cpp
  InferArgTypes.cpp
)

llvm_update_compile_flags(translate-llvm-to-std)

target_link_libraries(translate-llvm-to-std
  PRIVATE
  DynamaticSupport
  DynamaticHandshake
  DynamaticExperimentalSupport

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
  LLVMIRReader

  libclang
)

if(DYNAMATIC_ENABLE_XLS)
  target_link_libraries(dynamatic-opt
    PRIVATE
    XlsDialect
    DynamaticHandshakeToXls
  )
endif()
