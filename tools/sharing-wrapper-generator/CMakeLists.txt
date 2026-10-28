set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(exp-sharing-wrapper-generator sharing-wrapper-generator.cpp)
llvm_update_compile_flags(exp-sharing-wrapper-generator)
target_link_libraries(exp-sharing-wrapper-generator
  PRIVATE
  DynamaticSupport
  DynamaticHandshake
)
