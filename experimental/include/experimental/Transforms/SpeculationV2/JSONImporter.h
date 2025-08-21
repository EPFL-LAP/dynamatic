#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

struct SpecSpecification {
  llvm::SmallVector<unsigned> loopBBs;
};

mlir::FailureOr<SpecSpecification> readFromJSON(const std::string &jsonPath);
