add_llvm_executable(hls-fuzzer-check-bitwidth
  check-bitwidth.cpp
  PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(hls-fuzzer-check-bitwidth)
target_link_libraries(hls-fuzzer-check-bitwidth PRIVATE DynamaticHandshake MLIRParser)

add_dependencies(hls-fuzzer hls-fuzzer-check-bitwidth)

add_llvm_executable(hls-fuzzer-check-no-lsq
  check-no-lsq.cpp
  PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(hls-fuzzer-check-no-lsq)
target_link_libraries(hls-fuzzer-check-no-lsq PRIVATE DynamaticHandshake MLIRParser)

add_dependencies(hls-fuzzer hls-fuzzer-check-no-lsq)
