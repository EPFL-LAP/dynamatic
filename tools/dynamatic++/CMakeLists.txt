set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(dynamatic++
  dynamatic++.cpp
)

llvm_update_compile_flags(dynamatic++)
target_link_libraries(dynamatic++
  PRIVATE

  CIRCTESI
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTFIRRTL
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTHandshake
  CIRCTHandshakeToFIRRTL
  CIRCTHandshakeToHW
  CIRCTHandshakeTransforms
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTStandardToHandshake
  CIRCTSV
  CIRCTSVTransforms
  CIRCTTransforms
  
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRControlFlowDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFToControlFlow
  )
