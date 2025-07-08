set(LLVM_LINK_COMPONENTS
  Support
)

include_directories(${DYNAMATIC_SOURCE_DIR}/tools/visualizer-support)

add_llvm_executable(cex2csv cex2csv.cpp)
llvm_update_compile_flags(cex2csv)
target_link_libraries(cex2csv PRIVATE
  PRIVATE
  VisualizerSupport

  MLIRIR
  MLIRParser
  MLIRMemRefDialect

  DynamaticSupport
  DynamaticHandshake
  DynamaticAnalysis
)
