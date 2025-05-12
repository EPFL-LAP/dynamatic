set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(export-cfg
  export-cfg.cpp
)

llvm_update_compile_flags(export-cfg)
target_link_libraries(export-cfg
  PRIVATE
  DynamaticSupport
  DynamaticTransforms

  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  )
