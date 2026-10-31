# Include headers from tutorials
include_directories(${DYNAMATIC_SOURCE_DIR}/tutorials/CreatingPasses/include)
include_directories(${DYNAMATIC_BINARY_DIR}/tutorials/CreatingPasses/include)

# Include headers from the test passes
include_directories(${DYNAMATIC_SOURCE_DIR}/test/include)
include_directories(${DYNAMATIC_BINARY_DIR}/test/include)
include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/test/include)
include_directories(${DYNAMATIC_BINARY_DIR}/experimental/test/include)

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
  DynamaticLSQSizing
  DynamaticSpeculation
  DynamaticRigidification
  DynamaticFormalPropertyAnnotation
  DynamaticResourceSharing
  DynamaticCfToHandshake
  DynamaticFtdCfToHandshake
  DynamaticOperandRouting2CfToHandshake
  DynamaticHandshake
  DynamaticHandshakeToHW
  DynamaticHandshakeToSynth
  DynamaticBlifExporter
  DynamaticBlifImporter
  DynamaticTestTransforms
  DynamaticTransforms
  DynamaticTutorialsCreatingPasses
  DynamaticExperimentalTestTransforms
  DynamaticExperimentalSupport
  DynamaticExperimentalTransforms
  DynamaticExperimentalAnalysis
  DynamaticDuplication

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

if(DYNAMATIC_ENABLE_XLS)
  target_link_libraries(dynamatic-opt
    PRIVATE
    XlsDialect
    DynamaticHandshakeToXls
  )
endif()
