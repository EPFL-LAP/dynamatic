// RUN: dynamatic-opt %s --handshake-deactivate-mem-dependencies --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @test0(
handshake.func @test0(%arg0: !handshake.channel<i32>, %arg4: memref<8xi32>, %arg5: memref<8xi8>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["var1", "var0", "var2", "var0_start", "var2_start", "start"], resNames = ["var0_end", "var2_end", "end"]} {
  %0 = lsq[%arg5 : memref<8xi8>] (%arg7, %arg8, %addressResult_12, %dataResult_13, %arg8)  {groupSizes = [2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i8>, !handshake.control<>) -> !handshake.control<>
  %1:2 = lsq[%arg4 : memref<8xi32>] (%arg6, %arg8, %addressResult_10, %addressResult_14, %dataResult_15, %arg8)  {groupSizes = [2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
  %22 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>

// ^bb1:

  // CHECK: load
  // CHECK-SAME: #handshake<deps[{
  // CHECK-SAME: dstAccess : "store3"
  // CHECK-SAME: isActive : true
  // CHECK-SAME: }]
  %addressResult_10, %dataResult_11 = load[%22] %1#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[{dstAccess : "store3", loopDepth : 0, distance : 0, isActive : true}]>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
  %23 = trunci %dataResult_11 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
  %addressResult_12, %dataResult_13 = store[%22] %23 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[{dstAccess : "store0", loopDepth : 0, distance : 0, isActive : true}]>, handshake.name = "store2"} : <i32>, <i8>, <i32>, <i8>
  %addressResult_14, %dataResult_15 = store[%22] %arg0 {handshake.bb = 1 : ui32, handshake.name = "store3"} : <i32>, <i32>, <i32>, <i32>
  end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1#1, %0, %arg8 : <>, <>, <>
}

// -----

// CHECK-LABEL:   handshake.func @shrsi_giid(
handshake.func @shrsi_giid(%arg0: !handshake.channel<i8>, %arg1: memref<2xi16>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>) attributes {argNames = ["var5", "var1", "var1_start", "start"], resNames = ["var1_end", "end"]} {
  %0:2 = lsq[%arg1 : memref<2xi16>] (%arg2, %arg3, %addressResult, %addressResult_0, %dataResult_1, %arg3)  {groupSizes = [2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i16>, !handshake.control<>) -> (!handshake.channel<i16>, !handshake.control<>)
  %1 = constant %arg3 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
  // CHECK: load
  // CHECK-SAME: #handshake<deps[{
  // CHECK-SAME: dstAccess : "store1"
  // CHECK-SAME: isActive : false
  // CHECK-SAME: }]
  %addressResult, %dataResult = load[%1] %0#0 {handshake.bb = 0 : ui32, handshake.deps = #handshake<deps[{dstAccess : "store1", loopDepth : 0, distance : 0, isActive : true}]>, handshake.name = "load0"} : <i32>, <i16>, <i32>, <i16>
  %2 = extsi %dataResult {handshake.bb = 0 : ui32, handshake.name = "extsi0"} : <i16> to <i32>
  %3 = extui %arg0 {handshake.bb = 0 : ui32, handshake.name = "extui0"} : <i8> to <i32>
  %4 = shrsi %2, %3 {handshake.bb = 0 : ui32, handshake.name = "shrsi0"} : <i32>
  %5 = trunci %4 {handshake.bb = 0 : ui32, handshake.name = "trunci0"} : <i32> to <i16>
  %addressResult_0, %dataResult_1 = store[%1] %5 {handshake.bb = 0 : ui32, handshake.name = "store1"} : <i32>, <i16>, <i32>, <i16>
  end {handshake.bb = 0 : ui32, handshake.name = "end0"} %0#1, %arg3 : <>, <>
}
