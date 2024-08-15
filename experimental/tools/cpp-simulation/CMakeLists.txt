set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(cpp-simulation cpp-simulation.cpp)
llvm_update_compile_flags(cpp-simulation)
target_link_libraries(cpp-simulation PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticHandshake
  DynamaticHW
)