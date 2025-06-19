set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(rigidification-testbench
  rigidification-testbench.cpp
)

llvm_update_compile_flags(rigidification-testbench)
target_link_libraries(rigidification-testbench
  PRIVATE
  DynamaticHandshake
  DynamaticExperimentalSupport

  MLIRIR
  MLIRMemRefTransforms
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
)