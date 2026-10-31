// RUN: dynamatic-opt %s --split-input-file --verify-diagnostics

// Malformed groups of memory interfaces are rejected, by the parser for their
// syntax and by the verifier for their structure.

handshake.func @unknownAccessKind(%mem: memref<4xi32>, %memStart: <>, %a: <i2>, %ctrlEnd: <>) {
  // expected-error @below {{expected 'load' or 'store'}}
  %ldData, %memEnd = mem_controller[%mem : memref<4xi32>] start(%memStart) groups(bb0: [fetch(%a)]) end(%ctrlEnd) : <i2>, <i32>
  end
}

// -----

handshake.func @storeWithoutData(%mem: memref<4xi32>, %memStart: <>, %c: <i32>, %a: <i2>, %ctrlEnd: <>) {
  // expected-error @below {{expected ','}}
  %memEnd = mem_controller[%mem : memref<4xi32>] start(%memStart) groups(bb0 %c: [store(%a)]) end(%ctrlEnd) : <i2>, <i32>, <i32>
  end
}

// -----

handshake.func @badBlock(%mem: memref<4xi32>, %memStart: <>, %a: <i2>, %ctrlEnd: <>) {
  // expected-error @below {{expected a basic block (e.g., bb0)}}
  %ldData, %memEnd = mem_controller[%mem : memref<4xi32>] start(%memStart) groups(block0: [load(%a)]) end(%ctrlEnd) : <i2>, <i32>
  end
}

// -----

handshake.func @accessesWithoutBrackets(%mem: memref<4xi32>, %memStart: <>, %a: <i2>, %ctrlEnd: <>) {
  // expected-error @below {{expected '['}}
  %ldData, %memEnd = mem_controller[%mem : memref<4xi32>] start(%memStart) groups(bb0: load(%a)) end(%ctrlEnd) : <i2>, <i32>
  end
}

// -----

handshake.func @mcGroupWithoutBlock(%mem: memref<4xi32>, %memStart: <>, %c: <i32>, %a: <i2>, %d: <i32>, %ctrlEnd: <>) {
  // expected-error @below {{Every group of a memory controller must be a basic block.}}
  %memEnd = mem_controller[%mem : memref<4xi32>] start(%memStart) groups(%c: [store(%a, %d)]) end(%ctrlEnd) : <i2>, <i32>, <i32>
  end
}

// -----

handshake.func @duplicateBlock(%mem: memref<4xi32>, %memStart: <>, %a0: <i2>, %a1: <i2>, %ctrlEnd: <>) {
  // expected-error @below {{Block IDs must be unique but ID 0 was found at least twice in the list.}}
  %ldData:2, %memEnd = mem_controller[%mem : memref<4xi32>] start(%memStart) groups(bb0: [load(%a0)], bb0: [load(%a1)]) end(%ctrlEnd) : <i2>, <i32>
  end
}

// -----

handshake.func @mcStoreWithoutControl(%mem: memref<4xi32>, %memStart: <>, %a: <i2>, %d: <i32>, %ctrlEnd: <>) {
  // expected-error @below {{Block 0 stores, so it must have a control port.}}
  %memEnd = mem_controller[%mem : memref<4xi32>] start(%memStart) groups(bb0: [store(%a, %d)]) end(%ctrlEnd) : <i2>, <i32>
  end
}

// -----

handshake.func @lsqGroupWithoutControl(%mem: memref<4xi32>, %memStart: <>, %a: <i2>, %ctrlEnd: <>) {
  // expected-error @below {{Every LSQ group must have a control port.}}
  %ldData, %memEnd = lsq[%mem : memref<4xi32>] start(%memStart) groups(: [load(%a)]) end(%ctrlEnd) : <i2>, <i32>
  end
}

// -----

handshake.func @lsqEmptyGroup(%mem: memref<4xi32>, %memStart: <>, %a: <i2>, %g0: <>, %g1: <>, %ctrlEnd: <>) {
  // expected-error @below {{Every LSQ group must have at least one access.}}
  %ldData, %memEnd = lsq[%mem : memref<4xi32>] start(%memStart) groups(%g0: [load(%a)], %g1: []) end(%ctrlEnd) : <i2>, <i32>
  end
}

// -----

// The groups declare two loads, but the operands contain one (only possible in
// the generic form)
handshake.func @groupsMismatchOperands(%mem: memref<4xi32>, %memStart: <>, %a: <i2>, %ctrlEnd: <>) {
  // expected-error @below {{Expected 2 load address operands and data results.}}
  %ldData, %memEnd = "handshake.mem_controller"(%mem, %memStart, %a, %ctrlEnd) <{groups = [#handshake.mem_group<bb 0, ctrl = false, [load, load]>], operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0, 0, 0, 0, 1>, resultSegmentSizes = array<i32: 1, 0, 1>}> : (memref<4xi32>, !handshake.control<>, !handshake.channel<i2>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
  end
}
