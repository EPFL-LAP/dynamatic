set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(export-blif export-blif.cpp)
llvm_update_compile_flags(export-blif)
target_link_libraries(export-blif PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticSynth
  DynamaticHW
  DynamaticBlifExporter
)
