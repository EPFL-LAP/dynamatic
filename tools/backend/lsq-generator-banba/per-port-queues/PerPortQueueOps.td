#ifndef BANBA_PER_PORT_QUEUE_OPS_TD
#define BANBA_PER_PORT_QUEUE_OPS_TD

// A component of an LSQ with one queue per port.
// Its name in the IR starts with `per_port.`,
// and its C++ class is in the `banba::per_port` namespace
class PerPortComponent<string mnemonic, list<Trait> traits = []>
    : BanbaComponent<"per_port." # mnemonic, traits> {
  let cppNamespace = "::banba::per_port";
}

// Per-Port Group Allocator:
// manages entry allocation for an LSQ with one queue per port.
//
// Each group corresponds to a basic block,
// and so the allocation of a group is the allocation
// of one entry in the queue of each port
// used by a memory access in that basic block.
//
// A group is ready when each of those queues has a free entry.
//
// With several store queues,
// stores must still reach the memory in program order,
// and so the order in which groups are allocated
// is recorded in a FIFO, outside this component.
// A group is then only ready when that FIFO has room.
def PerPort_GroupAllocatorOp : PerPortComponent<"group_allocator"> {
  let arguments = (ins
      // clock and reset
      ClockType:$clk,
      I1:$rst,

      // the kernel's per-group init valid
      ArrayI1:$group_init_valid,

      // per queue, whether it has no free entry
      ArrayI1:$ld_full,
      ArrayI1:$st_full,

      // the write channel's group-order FIFO can take another group
      // (constant one with a single store queue, which needs no FIFO)
      I1:$st_order_ready);

  let results = (outs
      // the kernel's per-group init ready
      ArrayI1:$group_init_ready,

      // per group, whether it allocates this cycle
      ArrayI1:$group_allocate,

      // per queue, allocate one entry at the tail
      ArrayI1:$ld_wen,
      ArrayI1:$st_wen);

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Clock(),                             // clk
              t.Int(1),                              // rst
              t.Array(config.getNumBBs(), 1),        // group_init_valid
              t.Array(config.getNumLoadPorts(), 1),  // ld_full
              t.Array(config.getNumStorePorts(), 1), // st_full
              t.Int(1)};                             // st_order_ready
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getNumBBs(), 1),          // group_init_ready
              t.Array(config.getNumBBs(), 1),          // group_allocate
              t.Array(config.getNumLoadPorts(), 1),    // ld_wen
              t.Array(config.getNumStorePorts(), 1)};  // st_wen
    }
  }];
}

#endif // BANBA_PER_PORT_QUEUE_OPS_TD
