add_llvm_tool(buf-analysis
  buf-analysis.cpp
)

llvm_update_compile_flags(buf-analysis)
target_link_libraries(buf-analysis
  PRIVATE
  DynamaticSupport
  DynamaticTransforms

  DynamaticHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRMemRefDialect
  )
