#include "dynamatic/Support/LLVM.h"
#include <filesystem>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

// TODO move this somewhere else
int runNuXmv(const std::string &cmd, const std::string &stdoutFile);

// TODO move this somewhere else
LogicalResult createCMDfile(const std::filesystem::path &cmdPath,
                            const std::filesystem::path &smvPath,
                            const std::string &additionalCommands);

FailureOr<std::filesystem::path>
handshake2smv(const std::filesystem::path &mlirPath, bool png = false);

FailureOr<size_t> getSequenceLength(MLIRContext &context,
                                    const std::filesystem::path &outputDir,
                                    const std::string &mlirFile);
} // namespace dynamatic::experimental