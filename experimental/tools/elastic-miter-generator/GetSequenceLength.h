#include "dynamatic/Support/LLVM.h"

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// TODO move this somewhere else
int runNuXmv(const std::string &cmd, const std::string &stdoutFile);

LogicalResult handshake2smv(const std::string &mlirFilename, bool png = false);

FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::string &mlirFile);
} // namespace dynamatic::experimental