set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(rtl-constant-generator-verilog rtl-constant-generator-verilog.cpp)
llvm_update_compile_flags(rtl-constant-generator-verilog)
target_link_libraries(rtl-constant-generator-verilog
  PRIVATE
  DynamaticSupportRTL
  DynamaticHandshake
)
