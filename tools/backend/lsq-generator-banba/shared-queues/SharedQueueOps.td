#ifndef BANBA_SHARED_QUEUE_OPS_TD
#define BANBA_SHARED_QUEUE_OPS_TD

// A component of an LSQ whose queues are shared by all ports.
// Its name in the IR starts with `shared.`,
// and its C++ class is in the `banba::shared` namespace
class SharedComponent<string mnemonic, list<Trait> traits = []>
    : BanbaComponent<"shared." # mnemonic, traits> {
  let cppNamespace = "::banba::shared";
}

// Group Allocator:
// manages entry allocation for the Load-Store Queue (LSQ).
//
// Dynamatic's circuits have no encoding of program order,
// and so the group allocator unit of the LSQ has the
// program order of memory accesses in each basic block
// stored in Read-Only Memory (ROM).
//
// Each group corresponds to a basic block,
// and so the allocation of a group is the allocation
// of one queue entry per memory access in that basic block,
// and the updating of the order matrix to contain
// the order of those accesses, based on the order stored in ROM.
def Shared_GroupAllocatorOp : SharedComponent<"group_allocator"> {
  let arguments = (ins
      ClockType:$clk,
      I1:$rst,
      ArrayI1:$group_allocate_valid,
      Int:$ld_q_tail,
      Int:$ld_q_head,
      I1:$ld_q_empty,
      Int:$st_q_tail,
      Int:$st_q_head,
      I1:$st_q_empty);
  let results = (outs
      ArrayI1:$group_allocate_ready,
      ArrayI1:$ld_q_allocating,
      Int:$num_loads,
      ArrayI1:$st_q_allocating,
      Int:$num_stores,
      ArrayInt:$ga_ls_order,
      ArrayInt:$ld_q_port_idx,
      ArrayInt:$st_q_port_idx);

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      unsigned numGroups = config.getNumBBs();
      unsigned ldPointerWidth = config.getLdPointerWidth();
      unsigned stPointerWidth = config.getStPointerWidth();
      return {t.Clock(),               // clk
              t.Int(1),                // rst
              t.Array(numGroups, 1),   // group_allocate_valid
              t.Int(ldPointerWidth),   // ld_q_tail
              t.Int(ldPointerWidth),   // ld_q_head
              t.Int(1),                // ld_q_empty
              t.Int(stPointerWidth),   // st_q_tail
              t.Int(stPointerWidth),   // st_q_head
              t.Int(1)};               // st_q_empty
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      unsigned numGroups = config.getNumBBs();
      unsigned numLdEntries = config.getFifoDepth_L();
      unsigned numStEntries = config.getFifoDepth_S();
      return {t.Array(numGroups, 1),                               // group_allocate_ready
              t.Array(numLdEntries, 1),                            // ld_q_allocating
              t.Int(config.getLdCountWidth()),                     // num_loads
              t.Array(numStEntries, 1),                            // st_q_allocating
              t.Int(config.getStCountWidth()),                     // num_stores
              t.Array(numLdEntries, numStEntries),                 // ga_ls_order
              t.Array(numLdEntries, config.getLdPortIdxWidth()),   // ld_q_port_idx
              t.Array(numStEntries, config.getStPortIdxWidth())};  // st_q_port_idx
    }
  }];
}

#endif // BANBA_SHARED_QUEUE_OPS_TD
