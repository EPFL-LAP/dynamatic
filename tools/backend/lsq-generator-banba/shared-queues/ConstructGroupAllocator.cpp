#include "BanbaOps.h"
#include "BanbaSugar.h"
#include "llvm/Support/raw_ostream.h"

namespace banba::shared {

void GroupAllocatorOp::constructRegion() {
  llvm::outs() << "Construct shared group allocator op region\n";
}

} // namespace banba::shared

namespace banba::shared::ga {

void FreeEntryCalcOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.free_entry_calc op region\n";
}

void ReadyGenOp::constructRegion() {
  BanbaSugarScope scope(getOperation());
}

void PortIdxGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.port_idx_gen op region\n";
}

void OrderMatrixGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.order_matrix_gen op region\n";
}

void CountExtractOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.count_extract op region\n";
}

void AllocatingGenOp::constructRegion() {
  llvm::outs() << "Construct shared.ga.allocating_gen op region\n";
}

} // namespace banba::shared::ga
