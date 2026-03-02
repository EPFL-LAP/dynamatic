set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(import-blif import-blif.cpp)
llvm_update_compile_flags(import-blif)
target_link_libraries(import-blif PRIVATE
  PRIVATE   
  MLIRIR
  MLIRParser
  DynamaticSupport
  DynamaticSynth
  DynamaticHW
  DynamaticBlifImporter
)
