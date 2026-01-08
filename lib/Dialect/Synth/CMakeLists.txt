##===----------------------------------------------------------------------===//
## All the files related to the description of the Synth dialect 
## have been modified from the original version present in the 
## circt project at the following link:
## https://github.com/llvm/circt/tree/main/
##===----------------------------------------------------------------------===//

##===----------------------------------------------------------------------===//
##
## Implementation files for the Synth dialect.
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

