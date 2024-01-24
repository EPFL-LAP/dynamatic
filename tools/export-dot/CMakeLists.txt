set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(export-dot
  export-dot.cpp
)

llvm_update_compile_flags(export-dot)
target_link_libraries(export-dot
  PRIVATE
  DynamaticSupport
  DynamaticTransforms

  DynamaticHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  )
