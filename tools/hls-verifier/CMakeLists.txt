set(LLVM_LINK_COMPONENTS
  Support
)

include_directories(${DYNAMATIC_SOURCE_DIR}/tools/hls-verifier/include)

add_llvm_tool(hls-verifier
  hls-verifier.cpp
  lib/CAnalyser.cpp
  lib/Help.cpp
  lib/HlsLogging.cpp
  lib/HlsVhdlTb.cpp
  lib/Utilities.cpp
  lib/VerificationContext.cpp
)

llvm_update_compile_flags(hls-verifier)

find_package(Boost REQUIRED regex)
target_link_libraries(hls-verifier
  PRIVATE
  ${Boost_REGEX_LIBRARY}
  LLVMSupport
)
