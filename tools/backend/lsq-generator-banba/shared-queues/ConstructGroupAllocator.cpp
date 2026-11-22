#include "BanbaOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace banba;

void SharedGroupAllocatorOp::constructRegion() {
  llvm::outs() << "Construct shared group allocator op region\n";
}
