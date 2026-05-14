set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(no-short-circuit
  NoShortCircuit.cpp
)

llvm_update_compile_flags(no-short-circuit)
target_link_libraries(no-short-circuit
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  clangToolingCore
  clangToolingRefactoring
  clangTransformer
)
