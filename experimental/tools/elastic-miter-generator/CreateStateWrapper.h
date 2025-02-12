
#include "mlir/IR/BuiltinOps.h"
#include <llvm/ADT/StringSet.h>
#include <string>

using namespace mlir;

namespace dynamatic::experimental {

std::string createMiterCall(const SmallVector<std::string> &args,
                            const SmallVector<std::string> &res);

FailureOr<std::string> createStateWrapper(const std::string &smv, ModuleOp mlir,
                                          int n = 0, bool inf = false);

} // namespace dynamatic::experimental