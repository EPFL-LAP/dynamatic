#include "TargetRegistry.h"

std::unique_ptr<dynamatic::AbstractTarget>
dynamatic::TargetRegistry::getTarget(llvm::StringRef name) const {
  Constructor constructor = registry.lookup(name);
  if (!constructor)
    return nullptr;

  return constructor();
}
