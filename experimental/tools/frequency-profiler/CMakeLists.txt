add_llvm_executable(exp-frequency-profiler frequency-profiler.cpp Simulator.cpp)

llvm_update_compile_flags(exp-frequency-profiler)
target_link_libraries(exp-frequency-profiler PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  DynamaticExperimentalSupport
  )
