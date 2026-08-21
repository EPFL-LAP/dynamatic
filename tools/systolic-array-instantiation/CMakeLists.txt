# Link to required LLVM and Polly libraries
add_llvm_pass_plugin(SystolicArrayInstantiation
  SystolicArrayInstantiation.cpp
  DEPENDS
  LLVMCore
  LLVMSupport
  LLVMAnalysis
  Polly

  MLIRIR
  MLIRSupport
)
