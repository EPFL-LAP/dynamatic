add_library(LSQUsageAnalysisPass SHARED LSQUsageAnalysisPass.cpp)

# Link to required LLVM and Polly libraries
target_link_libraries(LSQUsageAnalysisPass
  PRIVATE LLVMCore LLVMSupport LLVMAnalysis Polly
)

# Required for plugins using new PassManager
target_compile_definitions(LSQUsageAnalysisPass PRIVATE LLVM_PLUGIN_EXPORT=1)