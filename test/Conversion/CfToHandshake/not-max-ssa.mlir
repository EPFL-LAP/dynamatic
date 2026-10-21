// RUN: dynamatic-opt --lower-cf-to-handshake %s --split-input-file --verify-diagnostics

// %0 is defined in ^bb0 but used in ^bb1 without being passed as a block
// argument.
func.func @notMaxSSA(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  cf.br ^bb1
^bb1:
  // expected-error @below {{operand #0 is defined in another block, but the region is expected to be in maximal SSA form (see --func-maximize-ssa)}}
  return %0 : i32
}
