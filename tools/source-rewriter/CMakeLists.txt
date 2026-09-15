set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
)

add_llvm_tool(source-rewriter
  SourceRewriter.cpp
)

llvm_update_compile_flags(source-rewriter)
if(TARGET clang-cpp)
  target_link_libraries(source-rewriter PRIVATE clang-cpp)
else()
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
endif()

# If building LLVM from source
# LLVM must be built before source-rewriter
if(DYNAMATIC_BUILD_LLVM)
  add_dependencies(source-rewriter clang-tablegen-targets)
endif()
