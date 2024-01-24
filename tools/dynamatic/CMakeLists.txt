set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(dynamatic
  dynamatic.cpp
)

llvm_update_compile_flags(dynamatic)
target_link_libraries(dynamatic
  PRIVATE
  DynamaticSupport
  DynamaticTransforms

  DynamaticHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRMemRefDialect
  )
