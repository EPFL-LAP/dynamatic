// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names --split-input-file | FileCheck %s

// An internal array, with a store and a load.
// It becomes a ram op, with the same interface as an argument memory
// CHECK-LABEL: handshake.func @internal_array
// the internal array gets a start argument and an end result, named ram0
// CHECK-SAME: argNames = ["in0", "ram0_start", "start"], resNames = ["out0", "ram0_end", "end"]
func.func @internal_array(%arg0: index {handshake.arg_name = "in0"}) -> i32 {
  // the internal array becomes a ram op, starting at all zeros
  // CHECK-DAG: %[[RAM:[^ ]+]] = "handshake.ram"() <{initialValue = dense<0> : memref<4xi32>}>
  %arr = memref.alloca() : memref<4xi32>
  %c42 = arith.constant 42 : i32
  memref.store %c42, %arr[%arg0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  %v = memref.load %arr[%arg0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  // the MC takes the memref of the ram op, and the start of ram0 (%arg1)
  // CHECK-DAG: %{{[^ ]+}}, %[[RAM0_END:[^ ]+]] = mem_controller[%[[RAM]] : memref<4xi32>] start(%arg1)
  // the end takes the loaded value, the end of the MC, then the start
  // CHECK-DAG: end {{.*}} %{{[^ ]+}}, %[[RAM0_END]], %arg2
  return %v : i32
}

// -----

// A constant global array, with a load.
// It becomes a ram op starting at the values of the global,
// with the same interface as an argument memory
memref.global "private" constant @table : memref<4xi32> = dense<[3, 1, 4, 1]>
// CHECK-LABEL: handshake.func @constant_global
// the global array gets a start argument and an end result, named ram0
// CHECK-SAME: argNames = ["in0", "ram0_start", "start"], resNames = ["out0", "ram0_end", "end"]
func.func @constant_global(%arg0: index {handshake.arg_name = "in0"}) -> i32 {
  // the global array becomes a ram op, starting at the values of the global
  // CHECK-DAG: %[[RAM:[^ ]+]] = "handshake.ram"() <{initialValue = dense<[3, 1, 4, 1]>
  %t = memref.get_global @table : memref<4xi32>
  %v = memref.load %t[%arg0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  // the MC takes the memref of the ram op, and the start of ram0 (%arg1)
  // CHECK-DAG: %{{[^ ]+}}, %[[RAM0_END:[^ ]+]] = mem_controller[%[[RAM]] : memref<4xi32>] start(%arg1)
  // the end takes the loaded value, the end of the MC, then the start
  // CHECK-DAG: end {{.*}} %{{[^ ]+}}, %[[RAM0_END]], %arg2
  return %v : i32
}
