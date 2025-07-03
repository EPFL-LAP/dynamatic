set(LLVM_LINK_COMPONENTS
  Support
)

include_directories(${DYNAMATIC_SOURCE_DIR}/tools/visualizer-support)
include_directories(${DYNAMATIC_SOURCE_DIR}/tools/hls-verifier/include)

add_llvm_executable(log2csv log2csv.cpp)
llvm_update_compile_flags(log2csv)
target_link_libraries(log2csv PRIVATE
  PRIVATE
  VisualizerSupport

  MLIRIR
  MLIRParser
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect

  DynamaticSupport
  DynamaticHandshake
  DynamaticAnalysis
)
