##===----------------------------------------------------------------------===//
##
## Implementation files for the Synth dialect.
##
##===----------------------------------------------------------------------===//
##
## This file originates from the CIRCT project (https://github.com/llvm/circt).
## It includes modifications made as part of Dynamatic.
##
##===----------------------------------------------------------------------===//

add_dynamatic_dialect_library(DynamaticSynth
  SynthDialect.cpp
  SynthOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${DYNAMATIC_MAIN_INCLUDE_DIR}/dynamatic/Dialect/Synth

  DEPENDS
  MLIRSynthIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  DynamaticHW
)

