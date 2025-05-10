set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(vivado-ip-adapter vivado-ip-adapter.cpp)
llvm_update_compile_flags(vivado-ip-adapter)
target_link_libraries(vivado-ip-adapter PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticSupportRTL
  DynamaticHandshake
  DynamaticHW
)
