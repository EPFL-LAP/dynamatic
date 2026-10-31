// RUN: dynamatic-opt %s --split-input-file | dynamatic-opt --split-input-file | FileCheck %s

// A memory controller groups its accesses by basic block, with a control port
// in the blocks that store. Block 2 has a control port but no access of its
// own, as its store goes through the LSQ, which forwards its accesses to the
// memory controller. A buffer between a store and the memory controller is
// legal, as the ports are identified by their operand segment.

// CHECK-LABEL: handshake.func @mcAndLSQ
// CHECK: mem_controller[%{{.*}} : memref<16xi32>] start(%{{.*}}) groups(bb0: [load(%{{.*}})], bb1 %{{.*}}: [store(%{{.*}}, %{{.*}}), load(%{{.*}})], bb2 %{{.*}}: []) lsq(%{{.*}}, %{{.*}}, %{{.*}}) end(%{{.*}}) : <i4>, <i32>, <i32>
// CHECK: lsq[MC] groups(%{{.*}}: [load(%{{.*}}), store(%{{.*}}, %{{.*}})]) mc(%{{.*}}) : <i4>, <i32>
handshake.func @mcAndLSQ(%mem: memref<16xi32>, %memStart: <>, %a0: <i4>, %a1: <i4>, %a2: <i4>, %a3: <i4>, %a4: <i4>, %d: <i32>, %d2: <i32>, %c1: <i32>, %c2: <i32>, %g: <>, %ctrlEnd: <>, %start: <>) -> (out0: <i32>, out1: <i32>, out2: <i32>, memEnd: <>, end: <>) {
  %ldData:2, %lsqLdData, %memEnd = mem_controller[%mem : memref<16xi32>] start(%memStart) groups(bb0: [load(%ldAddr0)], bb1 %c1: [store(%stAddrBuf, %stData1), load(%ldAddr1)], bb2 %c2: []) lsq(%ldAddrToMC, %stAddrToMC, %stDataToMC) end(%ctrlEnd) : <i4>, <i32>, <i32>
  %lsqLdData2, %ldAddrToMC, %stAddrToMC, %stDataToMC = lsq[MC] groups(%g: [load(%ldAddr2), store(%stAddr2, %stData2)]) mc(%lsqLdData) : <i4>, <i32>
  %ldAddr0, %v0 = load[%a0] %ldData#0 {handshake.bb = 0 : ui32} : <i4>, <i32>, <i4>, <i32>
  %stAddr1, %stData1 = store[%a1] %d {handshake.bb = 1 : ui32} : <i4>, <i32>, <i4>, <i32>
  %stAddrBuf = buffer %stAddr1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32} : <i4>
  %ldAddr1, %v1 = load[%a2] %ldData#1 {handshake.bb = 1 : ui32} : <i4>, <i32>, <i4>, <i32>
  %ldAddr2, %v2 = load[%a3] %lsqLdData2 {handshake.bb = 2 : ui32} : <i4>, <i32>, <i4>, <i32>
  %stAddr2, %stData2 = store[%a4] %d2 {handshake.bb = 2 : ui32} : <i4>, <i32>, <i4>, <i32>
  end {handshake.bb = 2 : ui32} %v0, %v1, %v2, %memEnd, %start : <i32>, <i32>, <i32>, <>, <>
}

// -----

// An LSQ that talks to the memory has its own start and end signals, and
// orders the accesses within each group.

// CHECK-LABEL: handshake.func @masterLSQ
// CHECK: lsq[%{{.*}} : memref<8xi16>] start(%{{.*}}) groups(%{{.*}}: [store(%{{.*}}, %{{.*}})], %{{.*}}: [load(%{{.*}}), load(%{{.*}})]) end(%{{.*}}) : <i3>, <i16>
handshake.func @masterLSQ(%mem: memref<8xi16>, %memStart: <>, %a0: <i3>, %a1: <i3>, %a2: <i3>, %d: <i16>, %g0: <>, %g1: <>, %ctrlEnd: <>, %start: <>) -> (out0: <i16>, out1: <i16>, memEnd: <>, end: <>) {
  %ldData:2, %memEnd = lsq[%mem : memref<8xi16>] start(%memStart) groups(%g0: [store(%stAddr, %stData)], %g1: [load(%ldAddr0), load(%ldAddr1)]) end(%ctrlEnd) : <i3>, <i16>
  %stAddr, %stData = store[%a0] %d {handshake.bb = 0 : ui32} : <i3>, <i16>, <i3>, <i16>
  %ldAddr0, %v0 = load[%a1] %ldData#0 {handshake.bb = 1 : ui32} : <i3>, <i16>, <i3>, <i16>
  %ldAddr1, %v1 = load[%a2] %ldData#1 {handshake.bb = 1 : ui32} : <i3>, <i16>, <i3>, <i16>
  end {handshake.bb = 1 : ui32} %v0, %v1, %memEnd, %start : <i16>, <i16>, <>, <>
}
