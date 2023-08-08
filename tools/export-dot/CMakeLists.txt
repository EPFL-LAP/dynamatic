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

  CIRCTHandshake
  CIRCTHandshakeTransforms

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRMemRefDialect
  )
