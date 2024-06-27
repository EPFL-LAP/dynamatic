set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(rtl-text-generator rtl-text-generator.cpp)
llvm_update_compile_flags(rtl-text-generator)
target_link_libraries(rtl-text-generator
  PRIVATE
  DynamaticSupport
  DynamaticHandshake
)
