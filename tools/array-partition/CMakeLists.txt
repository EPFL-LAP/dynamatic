add_library(ArrayPartition SHARED ArrayPartition.cpp)

# Link to required LLVM and Polly libraries
target_link_libraries(ArrayPartition
  PRIVATE
  LLVMCore
  LLVMSupport
  LLVMAnalysis
  Polly

  MLIRIR
  MLIRSupport

)

# Required for plugins using new PassManager
target_compile_definitions(ArrayPartition PRIVATE LLVM_PLUGIN_EXPORT=1)