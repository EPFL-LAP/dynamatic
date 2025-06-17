set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(rigidification
  rigidification.cpp
)

llvm_update_compile_flags(rigidification)
target_link_libraries(rigidification
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