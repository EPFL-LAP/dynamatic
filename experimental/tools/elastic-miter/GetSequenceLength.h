#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SEQUENCE_LENGTH_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SEQUENCE_LENGTH_H

#include "dynamatic/Support/LLVM.h"
#include <filesystem>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// Compute the length of the input sequence which is required to emulate an
// infinite sequence. This is done by enumerating the reachable states and then
// iteratively increasing the number of input tokens until the set of reachble
// states is equivalent.
FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::filesystem::path &outputDir,
                                    const std::filesystem::path &mlirPath);
} // namespace dynamatic::experimental

#endif