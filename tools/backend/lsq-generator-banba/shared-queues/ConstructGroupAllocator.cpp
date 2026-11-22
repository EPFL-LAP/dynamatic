#include "BanbaOps.h"
#include "llvm/Support/raw_ostream.h"

namespace banba::shared {

void GroupAllocatorOp::constructRegion() {
  llvm::outs() << "Construct shared group allocator op region\n";
}

} // namespace banba::shared
