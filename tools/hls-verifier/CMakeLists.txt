set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(hls-verifier
  hls-verifier.cpp
  src/CAnalyser.cpp
  src/Help.cpp
  src/HlsCoVerification.cpp
  src/HlsLogging.cpp
  src/HlsVhdlTb.cpp
  src/HlsVhdlVerification.cpp
  src/Utilities.cpp
  src/VerificationContext.cpp
)

llvm_update_compile_flags(hls-verifier)

find_package(Boost REQUIRED regex)
target_link_libraries(hls-verifier
  PRIVATE
  ${Boost_REGEX_LIBRARY}
  LLVMSupport
)
