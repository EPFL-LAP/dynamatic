// RUN: dynamatic-opt --func-maximize-ssa --exp-lower-cf-to-handshake %s --verify-diagnostics

// All accesses of an LSQ group must belong to the same basic block, since the
// group is allocated by the control signal of a single block.
func.func @lsqGroupMultipleBlocks(%mem: memref<64xi32>, %idx: index) -> i32 {
  %ldData = memref.load %mem[%idx] {handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  cf.br ^bb1
^bb1:
  // expected-error @below {{LSQ group 0 has memory accesses in multiple basic blocks; all accesses of an LSQ group must belong to the same block}}
  memref.store %ldData, %mem[%idx] {handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  return %ldData : i32
}
