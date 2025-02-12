
#include "mlir/IR/BuiltinOps.h"
#include <llvm/ADT/StringSet.h>
#include <string>

using namespace mlir;

namespace dynamatic::experimental {

std::string createMiterCall(const SmallVector<std::string> &args,
                            const SmallVector<std::string> &res);

FailureOr<std::string> createReachableStateWrapper(ModuleOp mlir, int n = 0,
                                                   bool inf = false);

FailureOr<std::string> createMiterWrapper(size_t bufferSize);

} // namespace dynamatic::experimental