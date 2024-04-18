set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(rtl-cmpi-generator rtl-cmpi-generator.cpp)
llvm_update_compile_flags(rtl-cmpi-generator)
target_link_libraries(rtl-cmpi-generator
  PRIVATE
  DynamaticSupport
  DynamaticHandshake
)
