set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Analysis)
add_public_tablegen_target(DynamaticExperimentalAnalysisPassIncGen)
add_dependencies(dynamatic-headers DynamaticExperimentalAnalysisPassIncGen)
