// RUN: dynamatic-opt %s --lower-cf-to-handshake --split-input-file --verify-diagnostics

// Two functions in one module
// expected-error @+1 {{cf to handshake supports a single function only}}
module {
  func.func @first(%arg0: i32 {handshake.arg_name = "in0"}) -> i32 {
    return %arg0 : i32
  }
  func.func @second(%arg0: i32 {handshake.arg_name = "in0"}) -> i32 {
    return %arg0 : i32
  }
}

// -----

// A function call
func.func private @callee(i32) -> i32
func.func @caller(%arg0: i32 {handshake.arg_name = "in0"}) -> i32 {
  // expected-error @+1 {{cf to handshake does not support function calls yet}}
  %0 = func.call @callee(%arg0) : (i32) -> i32
  return %0 : i32
}

// -----

// A global array which is not constant
// expected-error @+1 {{a global must be constant to become a RAM}}
memref.global "private" @table : memref<4xi32> = dense<[1, 2, 3, 4]>
func.func @non_constant_global(%arg0: index {handshake.arg_name = "in0"}) -> i32 {
  %0 = memref.get_global @table : memref<4xi32>
  %1 = memref.load %0[%arg0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  return %1 : i32
}
