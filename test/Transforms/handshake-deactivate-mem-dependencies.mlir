// RUN: dynamatic-opt %s --handshake-deactivate-mem-dependencies --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @test0(
handshake.func @test0(%var1: <i32>, %var0: memref<8xi32>, %var2: memref<8xi8>, %var0_start: <>, %var2_start: <>, %start: <>) -> (var0_end: <>, var2_end: <>, end: <>) {
  %0 = lsq[%var2 : memref<8xi8>] start(%var2_start) groups(%start: [store(%addressResult_12, %dataResult_13)]) end(%start) {handshake.name = "lsq0"} : <i32>, <i8>
  %1:2 = lsq[%var0 : memref<8xi32>] start(%var0_start) groups(%start: [load(%addressResult_10), store(%addressResult_14, %dataResult_15)]) end(%start) {handshake.name = "lsq1"} : <i32>, <i32>
  %22 = constant %start {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>

// ^bb1:

  // CHECK: = load[
  // CHECK-SAME: #handshake<deps[{
  // CHECK-SAME: dstAccess : "store3"
  // CHECK-SAME: isActive : true
  // CHECK-SAME: }]
  %addressResult_10, %dataResult_11 = load[%22] %1#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[{dstAccess : "store3", loopDepth : 0, distance : 0, isActive : true}]>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
  %23 = trunci %dataResult_11 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
  %addressResult_12, %dataResult_13 = store[%22] %23 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[{dstAccess : "store0", loopDepth : 0, distance : 0, isActive : true}]>, handshake.name = "store2"} : <i32>, <i8>, <i32>, <i8>
  %addressResult_14, %dataResult_15 = store[%22] %var1 {handshake.bb = 1 : ui32, handshake.name = "store3"} : <i32>, <i32>, <i32>, <i32>
  end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1#1, %0, %start : <>, <>, <>
}

// -----

// CHECK-LABEL:   handshake.func @shrsi_giid(
handshake.func @shrsi_giid(%var5: <i8>, %var1: memref<2xi16>, %var1_start: <>, %start: <>) -> (var1_end: <>, end: <>) {
  %0:2 = lsq[%var1 : memref<2xi16>] start(%var1_start) groups(%start: [load(%addressResult), store(%addressResult_0, %dataResult_1)]) end(%start) {handshake.name = "lsq0"} : <i32>, <i16>
  %1 = constant %start {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
  // CHECK: = load[
  // CHECK-SAME: #handshake<deps[{
  // CHECK-SAME: dstAccess : "store1"
  // CHECK-SAME: isActive : false
  // CHECK-SAME: }]
  %addressResult, %dataResult = load[%1] %0#0 {handshake.bb = 0 : ui32, handshake.deps = #handshake<deps[{dstAccess : "store1", loopDepth : 0, distance : 0, isActive : true}]>, handshake.name = "load0"} : <i32>, <i16>, <i32>, <i16>
  %2 = extsi %dataResult {handshake.bb = 0 : ui32, handshake.name = "extsi0"} : <i16> to <i32>
  %3 = extui %var5 {handshake.bb = 0 : ui32, handshake.name = "extui0"} : <i8> to <i32>
  %4 = shrsi %2, %3 {handshake.bb = 0 : ui32, handshake.name = "shrsi0"} : <i32>
  %5 = trunci %4 {handshake.bb = 0 : ui32, handshake.name = "trunci0"} : <i32> to <i16>
  %addressResult_0, %dataResult_1 = store[%1] %5 {handshake.bb = 0 : ui32, handshake.name = "store1"} : <i32>, <i16>, <i32>, <i16>
  end {handshake.bb = 0 : ui32, handshake.name = "end0"} %0#1, %start : <>, <>
}
