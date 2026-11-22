#include "BanbaOps.h"
#include "llvm/Support/raw_ostream.h"

namespace banba::per_port {

void GroupAllocatorOp::constructRegion() {
  llvm::outs() << "Construct per-port group allocator op region\n";
}

} // namespace banba::per_port
