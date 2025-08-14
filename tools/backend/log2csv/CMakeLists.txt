set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(log2csv log2csv.cpp)
llvm_update_compile_flags(log2csv)
target_link_libraries(log2csv PRIVATE
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect

  DynamaticSupport
  DynamaticHandshake
  DynamaticAnalysis
)
