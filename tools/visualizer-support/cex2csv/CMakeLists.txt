set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(cex2csv cex2csv.cpp)
llvm_update_compile_flags(cex2csv)
target_link_libraries(cex2csv PRIVATE
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRMemRefDialect

  DynamaticSupport
  DynamaticHandshake
  DynamaticAnalysis
)
