set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(smv-generator smv-generator.cpp)
llvm_update_compile_flags(smv-generator)
target_link_libraries(smv-generator
  PRIVATE
  DynamaticSupportRTL
  DynamaticHandshake
)
