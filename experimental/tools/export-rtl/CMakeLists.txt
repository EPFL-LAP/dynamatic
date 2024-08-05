set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(exp-export-rtl export-rtl.cpp)
llvm_update_compile_flags(exp-export-rtl)
target_link_libraries(exp-export-rtl PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupportRTL
  DynamaticHandshake
  DynamaticHW
)
