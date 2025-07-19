add_library(DependenceAnalysisPass SHARED DependenceAnalysisPass.cpp)


# Link to required LLVM and Polly libraries
target_link_libraries(DependenceAnalysisPass
  PRIVATE
  LLVMCore
  LLVMIRReader
  LLVMAnalysis
  LLVMPasses
  LLVMFrontendOpenMP

  Polly # The main Polly library
)

# Required for plugins using new PassManager
target_compile_definitions(DependenceAnalysisPass PRIVATE LLVM_PLUGIN_EXPORT=1)