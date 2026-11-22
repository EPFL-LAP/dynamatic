#include "BanbaOps.h"
#include "BanbaSugar.h"
#include "llvm/Support/raw_ostream.h"

namespace banba::shared {

void GroupAllocatorOp::constructRegion() {
  llvm::outs() << "Construct shared group allocator op region\n";
}

} // namespace banba::shared

namespace banba::shared::ga {

void FreeEntryOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.free_entry op region\n";
}

void ReadyOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
}

void PortIdxGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.port_idx_gen op region\n";
}

void OrderMatrixOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.order_matrix op region\n";
}

void CountsOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.counts op region\n";
}

void AllocatingGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.allocating_gen op region\n";
}

} // namespace banba::shared::ga
