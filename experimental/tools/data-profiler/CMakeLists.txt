add_llvm_executable(data-profiler
  data-profiler.cpp
  Simulator.cpp
)

llvm_update_compile_flags(data-profiler)
target_link_libraries(data-profiler PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  DynamaticSupport 
  DynamaticExperimentalSupport
  )