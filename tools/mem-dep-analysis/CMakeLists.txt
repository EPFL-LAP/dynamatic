set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(mem-dep-analysis
  Driver.cpp
  MemDepAnalysis.cpp
)

llvm_update_compile_flags(mem-dep-analysis)

target_compile_options(mem-dep-analysis PRIVATE
    -fsanitize=address
)

# Link to required LLVM and Polly libraries
target_link_libraries(mem-dep-analysis
  PRIVATE
  DynamaticSupport
  DynamaticHandshake
  DynamaticExperimentalSupport

  Polly

  LLVMAnalysis
  LLVMBitReader
  LLVMCore
  LLVMIRReader
  LLVMPasses
  LLVMSupport

  libclang
)

