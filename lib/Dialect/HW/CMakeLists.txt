add_dynamatic_dialect_library(DynamaticHW
  ConversionPatterns.cpp
  CustomDirectiveImpl.cpp
  HWAttributes.cpp
  HWDialect.cpp
  HWInstanceImplementation.cpp
  HWModuleOpInterface.cpp
  HWOpInterfaces.cpp
  HWOps.cpp
  HWTypeInterfaces.cpp
  HWTypes.cpp
  InnerSymbolTable.cpp
  ModuleImplementation.cpp
  
  DEPENDS
  MLIRHWIncGen
  MLIRHWAttrIncGen
  MLIRHWEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
)
