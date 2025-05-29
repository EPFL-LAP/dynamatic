set(LLVM_LINK_COMPONENTS
  Support
)

include_directories(${DYNAMATIC_SOURCE_DIR}/tools/hls-verifier/include)

add_llvm_tool(hls-verifier
  hls-verifier.cpp
  lib/Help.cpp
  lib/HlsLogging.cpp
  lib/HlsVhdlTb.cpp
  lib/Utilities.cpp
)

llvm_update_compile_flags(hls-verifier)

target_link_libraries(hls-verifier PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticSupportRTL
  DynamaticHandshake
  DynamaticHW
)

target_link_libraries(hls-verifier
  PRIVATE
  LLVMSupport
)
