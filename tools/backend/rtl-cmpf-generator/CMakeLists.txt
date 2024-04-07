set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(rtl-cmpf-generator rtl-cmpf-generator.cpp)
llvm_update_compile_flags(rtl-cmpf-generator)
target_link_libraries(rtl-cmpf-generator
  PRIVATE
  DynamaticSupport
  DynamaticHandshake
)
