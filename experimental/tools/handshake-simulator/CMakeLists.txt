set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(handshake-simulator handshake-simulator.cpp)
llvm_update_compile_flags(handshake-simulator)
target_link_libraries(handshake-simulator PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticHandshake
  DynamaticHW
  DynamaticExperimentalSupport
)
