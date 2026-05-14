set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(source-rewriter
  SourceRewriter.cpp
)

llvm_update_compile_flags(source-rewriter)
target_link_libraries(source-rewriter
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
