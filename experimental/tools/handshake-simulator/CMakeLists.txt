add_llvm_executable(handshake-simulator handshake-simulator.cpp Simulation.cpp)

llvm_update_compile_flags(handshake-simulator)
target_link_libraries(handshake-simulator PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport

  CIRCTHandshake
  DynamaticSupport
  CIRCTStandardToHandshake
  )
