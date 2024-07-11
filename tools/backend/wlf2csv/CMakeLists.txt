set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(wlf2csv wlf2csv.cpp)
llvm_update_compile_flags(wlf2csv)
target_link_libraries(wlf2csv PRIVATE
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
