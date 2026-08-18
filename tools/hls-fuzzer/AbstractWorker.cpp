#include "AbstractWorker.h"

#include "llvm/ADT/STLExtras.h"

dynamatic::AbstractWorker::~AbstractWorker() = default;

bool dynamatic::AbstractWorker::isStatisticEnabled(llvm::StringRef name) const {
  if (!options.statistics)
    return false;

  // An empty list means all statistics are enabled.
  return options.statistics->empty() ||
         llvm::is_contained(*options.statistics, name);
}
