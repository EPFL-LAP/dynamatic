add_llvm_executable(hls-fuzzer-check-bitwidth
  check-bitwidth.cpp
)
llvm_update_compile_flags(hls-fuzzer-check-bitwidth)
target_link_libraries(hls-fuzzer-check-bitwidth PRIVATE DynamaticHandshake MLIRParser)

add_dependencies(hls-fuzzer hls-fuzzer-check-bitwidth)
