set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(elastic-miter
  elastic-miter.cpp
  GetSequenceLength.cpp
  FabricGeneration.cpp
  ElasticMiterTestbench.cpp
  SmvUtils.cpp
  Constraints.cpp
)

llvm_update_compile_flags(elastic-miter)
target_link_libraries(elastic-miter
  PRIVATE
  DynamaticHandshake
  DynamaticSupport
  DynamaticTransforms
  DynamaticExperimentalSupport

  MLIRIR
  MLIRMemRefTransforms
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
)
