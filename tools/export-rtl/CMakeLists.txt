set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(export-rtl export-rtl.cpp)
llvm_update_compile_flags(export-rtl)
target_link_libraries(export-rtl PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticSupportRTL
  DynamaticHandshake
  DynamaticHW
  DynamaticExperimentalSupport
)
