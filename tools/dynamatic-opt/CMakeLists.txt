set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(dynamatic-opt
  dynamatic-opt.cpp
)

llvm_update_compile_flags(dynamatic-opt)
target_link_libraries(dynamatic-opt
  PRIVATE
  AffineToScf
  DynamaticStandardToHandshakeFPGA18
  DynamaticExportDOT
  DynamaticHandshakeToNetlist
  DynamaticTransforms
  DynamaticTutorialsCreatingPasses
  DynamaticExperimentalTestTransforms

  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTStandardToHandshake

  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect
)
