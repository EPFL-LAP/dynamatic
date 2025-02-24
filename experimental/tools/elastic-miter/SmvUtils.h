#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SMV_UTILS_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_SMV_UTILS_H

#include "mlir/Support/LogicalResult.h"
#include <filesystem>
#include <string>

using namespace mlir;
using namespace llvm;

namespace dynamatic::experimental {

LogicalResult createCMDfile(const std::filesystem::path &cmdPath,
                            const std::filesystem::path &smvPath,
                            const std::string &additionalCommands);

int runNuXmv(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile);

int runNuSMV(const std::filesystem::path &cmdPath,
             const std::filesystem::path &stdoutFile);

int runSmvCmd(const std::filesystem::path &cmdPath,
              const std::filesystem::path &stdoutFile);

FailureOr<std::filesystem::path>
handshake2smv(const std::filesystem::path &mlirPath,
              const std::filesystem::path &outputDir, bool png = false);

} // namespace dynamatic::experimental
#endif