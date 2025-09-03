# Link to required LLVM and Polly libraries
add_llvm_pass_plugin(MemDepAnalysis
  MemDepAnalysis.cpp
  DEPENDS
  LLVMCore
  LLVMSupport
  LLVMAnalysis
  Polly

  MLIRIR
  MLIRSupport
)
