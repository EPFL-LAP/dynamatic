set(LLVM_OPTIONAL_SOURCES
  MemDepAnalysis.cpp
  ArrayPartition.cpp
)


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

# Link to required LLVM and Polly libraries
add_llvm_pass_plugin(ArrayPartition
  ArrayPartition.cpp
  DEPENDS
  LLVMCore
  LLVMSupport
  LLVMAnalysis
  Polly

  MLIRIR
  MLIRSupport
)
