add_dynamatic_library(DynamaticLLVMToControlFlow
  LLVMToControlFlow.cpp

  DEPENDS
  DynamaticConversionPassIncGen

  LINK_LIBS PUBLIC
  DynamaticSupport
  DynamaticHandshake
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIndexDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard

  MLIRLLVMDialect
  MLIRArithDialect
  )
