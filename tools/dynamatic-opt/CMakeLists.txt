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
  DynamaticLowerScfToCf
  DynamaticBufferPlacement
  DynamaticStandardToHandshakeFPGA18
  DynamaticHandshakeToNetlist
  DynamaticTransforms
  DynamaticTutorialsCreatingPasses
  DynamaticExperimentalTestTransforms
  DynamaticExperimentalSupport
  DynamaticExperimentalTransforms

  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTStandardToHandshake

  MLIRIR
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRSCFTransforms
  MLIRMemRefTransforms
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect
)
