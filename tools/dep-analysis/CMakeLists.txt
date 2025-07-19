add_library(DependenceAnalysisPass SHARED DependenceAnalysisPass.cpp)

# Link to required LLVM and Polly libraries
target_link_libraries(DependenceAnalysisPass
  PRIVATE LLVMCore LLVMSupport LLVMAnalysis Polly
)

# Required for plugins using new PassManager
target_compile_definitions(DependenceAnalysisPass PRIVATE LLVM_PLUGIN_EXPORT=1)