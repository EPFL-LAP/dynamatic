set(LLVM_LINK_COMPONENTS
  Support
)
add_llvm_executable(exp-export-smv export-smv.cpp)
llvm_update_compile_flags(exp-export-smv)
target_link_libraries(exp-export-smv
  PRIVATE
  DynamaticSupport
  DynamaticTransforms

  DynamaticHandshake
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRLLVMDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
)