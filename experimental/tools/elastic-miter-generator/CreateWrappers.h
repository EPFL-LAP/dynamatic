
#include "mlir/IR/BuiltinOps.h"
#include <filesystem>
#include <llvm/ADT/StringSet.h>
#include <string>

using namespace mlir;

namespace dynamatic::experimental {

std::string createMiterCall(const SmallVector<std::string> &args,
                            const SmallVector<std::string> &res);

FailureOr<std::string> createReachableStateWrapper(ModuleOp mlir, int n = 0,
                                                   bool inf = false);

LogicalResult createMiterWrapper(const std::filesystem::path &wrapperPath,
                                 const std::filesystem::path &jsonPath,
                                 const std::string &modelSmvName,
                                 size_t nrOfTokens);

} // namespace dynamatic::experimental