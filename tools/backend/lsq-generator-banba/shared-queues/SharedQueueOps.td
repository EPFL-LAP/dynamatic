#ifndef BANBA_SHARED_QUEUE_OPS_TD
#define BANBA_SHARED_QUEUE_OPS_TD

// A component of an LSQ whose queues are shared by all ports.
// Its name in the IR starts with `shared.`,
// and its C++ class is in the `banba::shared` namespace
class SharedComponent<string mnemonic, list<Trait> traits = []>
    : ConfigParamComponent<"shared." # mnemonic, traits> {
  let cppNamespace = "::banba::shared";
}

def Shared_GroupAllocatorOp : SharedComponent<"group_allocator"> {
  let arguments = (ins
      ClockType:$clk,
      I1:$rst,
      ArrayI1:$ga_valid,
      Int:$ld_q_tail,
      Int:$ld_q_head,
      I1:$ld_q_empty,
      Int:$st_q_tail,
      Int:$st_q_head,
      I1:$st_q_empty);
  let results = (outs
      ArrayI1:$ga_ready,
      ArrayI1:$ld_q_allocating,
      Int:$num_loads,
      ArrayI1:$st_q_allocating,
      Int:$num_stores,
      ArrayInt:$ga_ls_order,
      ArrayInt:$ld_q_port_idx,
      ArrayInt:$st_q_port_idx);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "Group Allocator:",
        "manages entry allocation for the Load-Store Queue (LSQ).",
        "",
        "Dynamatic's circuits have no encoding of program order,",
        "and so the group allocator unit of the LSQ has the",
        "program order of memory accesses in each basic block",
        "stored in Read-Only Memory (ROM).",
        "",
        "Each group corresponds to a basic block,",
        "and so the allocation of a group is the allocation",
        "of one queue entry per memory access",
        "in that basic block,",
        "and the updating of the order matrix to contain",
        "the order of those accesses,",
        "based on the order stored in ROM.",
        "",
        "Sub-components:",
        "",
        "1. Free Entry Calculation:",
        "   the number of free entries in each queue,",
        "   from its head and tail pointers,",
        "",
        "2. Ready Generation:",
        "   checks if there is sufficient space in the queues",
        "   to allocate a group",
        "",
        "3. Port Index Generation:",
        "   drives the port index inputs of both queues",
        "   so that the newly allocated queue entries",
        "   know which port they belong to",
        "",
        "4. Order Matrix Generation:",
        "   Generates a partial order matrix",
        "   representing only the newly allocated group,",
        "   already shifted based on the queue pointers",
        "   so it can be easily combined",
        "   with the current order matrix",
        "",
        "5. Load/Store Count Extraction:",
        "   Shallow wrapper that reads the number",
        "   of loads and stores of the group being allocated",
        "   from ROM",
        "",
        "6. Allocating Generation:",
        "   Drives the per-entry allocating input of each queue"};
  }];

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      unsigned numGroups = config.getNumBBs();
      unsigned ldPointerWidth = config.getLdPointerWidth();
      unsigned stPointerWidth = config.getStPointerWidth();
      return {t.Clock(),               // clk
              t.Int(1),                // rst
              t.Array(numGroups, 1),   // ga_valid
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
      return {t.Array(numGroups, 1),                               // ga_ready
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

// A sub-component of the shared group allocator.
// Its name in the IR starts with `shared.ga.`,
// and its C++ class is in the `banba::shared::ga` namespace
class SharedGaComponent<string mnemonic, list<Trait> traits = []>
    : SharedComponent<"ga." # mnemonic, traits> {
  let cppNamespace = "::banba::shared::ga";
}

def SharedGa_FreeEntryCalcOp : SharedGaComponent<"free_entry_calc"> {
  let arguments = (ins
      Int:$ld_q_head,
      Int:$ld_q_tail,
      Int:$st_q_head,
      Int:$st_q_tail,
      I1:$ld_q_empty,
      I1:$st_q_empty);
  let results = (outs
      Int:$ld_num_free_entries,
      Int:$st_num_free_entries);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "Free Entry Calculation:",
        "Calculates the number of free entries in each queue,",
        "by cyclic subtraction (WrapSub) of the head and tail pointers.",
        "The subtraction is ambiguous when head == tail (empty or full).",
        "We use a mux on the queue's empty flag to resolve it,",
        "outputting the maximum queue size when empty",
        "and the WrapSub result otherwise."};
  }];

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      unsigned ldPointerWidth = config.getLdPointerWidth();
      unsigned stPointerWidth = config.getStPointerWidth();
      return {t.Int(ldPointerWidth),   // ld_q_head
              t.Int(ldPointerWidth),   // ld_q_tail
              t.Int(stPointerWidth),   // st_q_head
              t.Int(stPointerWidth),   // st_q_tail
              t.Int(1),                // ld_q_empty
              t.Int(1)};               // st_q_empty
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Int(config.getLdFreeEntriesWidth()),   // ld_num_free_entries
              t.Int(config.getStFreeEntriesWidth())};  // st_num_free_entries
    }
  }];
}

def SharedGa_ReadyGenOp : SharedGaComponent<"ready_gen"> {
  let arguments = (ins
      Int:$ld_num_free_entries,
      Int:$st_num_free_entries);
  let results = (outs ArrayI1:$ga_ready);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "Ready Signal Generation:",
        "for each group, compare available space in each queue",
        "with the required space to allocate that group.",
        "A group is ready only if there is enough space",
        "for both its loads and its stores."};
  }];

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Int(config.getLdFreeEntriesWidth()),   // ld_num_free_entries
              t.Int(config.getStFreeEntriesWidth())};  // st_num_free_entries
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getNumBBs(), 1)};  // ga_ready
    }
  }];
}

def SharedGa_PortIdxGenOp : SharedGaComponent<"port_idx_gen"> {
  let arguments = (ins
      ArrayI1:$group_allocate,
      Int:$ld_q_tail,
      Int:$st_q_tail);
  let results = (outs
      ArrayInt:$ld_q_port_idx,
      ArrayInt:$st_q_port_idx);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "Port Index Generation:",
        "ROM-lookup the source port of each entry",
        "for the allocated group,",
        "then CyclicLeftShift by each queue's tail to align it."};
  }];

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getNumBBs(), 1),      // group_allocate
              t.Int(config.getLdPointerWidth()),   // ld_q_tail
              t.Int(config.getStPointerWidth())};  // st_q_tail
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getFifoDepth_L(),
                      config.getLdPortIdxWidth()),   // ld_q_port_idx
              t.Array(config.getFifoDepth_S(),
                      config.getStPortIdxWidth())};  // st_q_port_idx
    }
  }];
}

def SharedGa_OrderMatrixGenOp : SharedGaComponent<"order_matrix_gen"> {
  let arguments = (ins
      ArrayI1:$group_allocate,
      Int:$ld_q_tail,
      Int:$st_q_tail);
  let results = (outs ArrayInt:$ga_ls_order);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "Order Matrix Generation:",
        "ROM lookup of the order inside",
        "the allocated group,",
        "first builds an un-shifted partial load-store order matrix,",
        "then left shifts it twice, by st_q_tail and ld_q_tail,",
        "to match the partial order matrix",
        "with the LSQ's main order matrix."};

    /// The comment printed in its place
    /// when no group has a store before a load
    static constexpr StringRef ZeroModComment[] = {
        "Order Matrix Generation:",
        "In this kernel,",
        "There is never a store before a load inside a basic block"};
  }];

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getNumBBs(), 1),      // group_allocate
              t.Int(config.getLdPointerWidth()),   // ld_q_tail
              t.Int(config.getStPointerWidth())};  // st_q_tail
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getFifoDepth_L(),
                      config.getFifoDepth_S())};  // ga_ls_order
    }
  }];
}

def SharedGa_CountExtractOp : SharedGaComponent<"count_extract"> {
  let arguments = (ins ArrayI1:$group_allocate);
  let results = (outs
      Int:$num_loads,
      Int:$num_stores);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "Load/Store Count Extraction:",
        "the number of loads and stores in the allocated group,",
        "from a ROM lookup by the one-hot group_allocate."};

    /// The comment printed in its place when there is a single group
    static constexpr StringRef SingleGroupModComment[] = {
        "Load/Store Count Extraction:",
        "the number of loads and stores in the allocated group.",
        "There is a single group, so each count is that group's",
        "when it allocates, and zero otherwise."};
  }];

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getNumBBs(), 1)};  // group_allocate
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Int(config.getLdCountWidth()),   // num_loads
              t.Int(config.getStCountWidth())};  // num_stores
    }
  }];
}

def SharedGa_AllocatingGenOp : SharedGaComponent<"allocating_gen"> {
  let arguments = (ins
      Int:$num_loads,
      Int:$num_stores,
      Int:$ld_q_tail,
      Int:$st_q_tail);
  let results = (outs
      ArrayI1:$ld_q_allocating,
      ArrayI1:$st_q_allocating);

  let extraClassDeclaration = [{
    /// The comment printed above the operation's module in the SystemVerilog
    static constexpr StringRef ModComment[] = {
        "Allocating:",
        "the per-entry allocation masks for both queues.",
        "For each queue, create an unshifted bitmask,",
        "then CyclicLeftShift it by the queue tail pointer",
        "to rotate the block of 1s to start at the tail position."};
  }];

  let extraClassDefinition = [{
    SmallVector<Type> $cppClass::getExpectedOperandTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Int(config.getLdCountWidth()),     // num_loads
              t.Int(config.getStCountWidth()),     // num_stores
              t.Int(config.getLdPointerWidth()),   // ld_q_tail
              t.Int(config.getStPointerWidth())};  // st_q_tail
    }
    SmallVector<Type> $cppClass::getExpectedResultTypes(ConfigAttr config) {
      Types t(config.getContext());
      return {t.Array(config.getFifoDepth_L(), 1),   // ld_q_allocating
              t.Array(config.getFifoDepth_S(), 1)};  // st_q_allocating
    }
  }];
}

#endif // BANBA_SHARED_QUEUE_OPS_TD
