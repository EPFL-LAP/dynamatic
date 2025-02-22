#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SEQUENCE_LENGTH_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SEQUENCE_LENGTH_H

#include "dynamatic/Support/LLVM.h"
#include <filesystem>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

int compareReachableStates(const std::string &infFile,
                           const std::string &finFile,
                           const std::string &modelName);

FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::filesystem::path &outputDir,
                                    const std::string &mlirFile);
} // namespace dynamatic::experimental
#endif