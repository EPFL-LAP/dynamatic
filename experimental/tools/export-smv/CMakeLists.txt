set(LLVM_LINK_COMPONENTS
  Support
)
add_llvm_executable(export-smv export-smv.cpp)
llvm_update_compile_flags(export-smv)
target_link_libraries(export-smv
  PRIVATE
  DynamaticSupport
  DynamaticTransforms
  MLIRIR
  DynamaticSupportRTL
  DynamaticHW

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