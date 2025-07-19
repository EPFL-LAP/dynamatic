add_library(MemDepAnalysis SHARED MemDepAnalysis.cpp)

# Link to required LLVM and Polly libraries
target_link_libraries(MemDepAnalysis
  PRIVATE
  LLVMCore
  LLVMSupport
  LLVMAnalysis
  Polly

  MLIRIR
  MLIRSupport
)

# Required for plugins using new PassManager
target_compile_definitions(MemDepAnalysis PRIVATE LLVM_PLUGIN_EXPORT=1)