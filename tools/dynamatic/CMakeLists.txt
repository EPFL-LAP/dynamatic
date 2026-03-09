set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(dynamatic
  dynamatic.cpp
  DEPENDS
  # Required by compile.sh
  dynamatic-opt opt clang MemDepAnalysis ArrayPartition translate-llvm-to-std
  export-dot export-cfg exp-frequency-profiler
  # Required by simulate.sh
  hls-verifier
  # Required by write-hdl.sh
  export-rtl
)

llvm_update_compile_flags(dynamatic)
target_link_libraries(dynamatic
  PRIVATE
  DynamaticSupport
  DynamaticTransforms

  DynamaticHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRMemRefDialect
  readline
  )
