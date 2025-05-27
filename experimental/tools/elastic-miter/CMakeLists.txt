set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(elastic-miter
  elastic-miter.cpp
  GetSequenceLength.cpp
  FabricGeneration.cpp
  CreateSmvFormalTestbench.cpp
  SmvUtils.cpp
  Constraints.cpp
)

llvm_update_compile_flags(elastic-miter)
target_link_libraries(elastic-miter
  PRIVATE
  DynamaticBufferPlacement
  DynamaticLSQSizing
  DynamaticExperimentalResourceSharing
  DynamaticHandshake
  DynamaticTestTransforms
  DynamaticTransforms
  DynamaticTutorialsCreatingPasses
  DynamaticExperimentalTestTransforms
  DynamaticExperimentalSupport
  DynamaticExperimentalTransforms

  MLIRIR
  MLIRArithTransforms
  MLIRMemRefTransforms
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
)
