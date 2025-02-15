#include "dynamatic/Support/LLVM.h"
#include <filesystem>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// TODO move this somewhere else
int runNuXmv(const std::string &cmd, const std::string &stdoutFile);

FailureOr<std::filesystem::path>
handshake2smv(const std::filesystem::path &mlirPath, bool png = false);

FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::string &mlirFile);
} // namespace dynamatic::experimental