// RUN: dynamatic-opt %s --lower-cf-to-handshake --split-input-file --verify-diagnostics

// A memory access without a name
func.func @unnamed_access(%arg0: memref<4xi32> {handshake.arg_name = "mem0"}, %arg1: index {handshake.arg_name = "in0"}) -> i32 {
  // expected-error @+1 {{memory access has no name}}
  %0 = memref.load %arg0[%arg1] {handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  return %0 : i32
}

// -----

// A named op which is not a memory access
func.func @named_non_access(%arg0: i32 {handshake.arg_name = "in0"}) -> i32 {
  // expected-error @+1 {{only memory accesses should be named}}
  %0 = arith.addi %arg0, %arg0 {handshake.name = "add0"} : i32
  return %0 : i32
}
