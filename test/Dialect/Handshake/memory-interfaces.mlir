// RUN: dynamatic-opt %s --split-input-file | dynamatic-opt --split-input-file | FileCheck %s

// A memory controller groups its accesses by basic block, with a control port
// in the blocks that store. Block 2 has a control port but no access of its
// own, as its store goes through the LSQ, which forwards its accesses to the
// memory controller. A buffer between a store and the memory controller is
// legal, as the ports are identified by their operand segment.

// CHECK-LABEL: handshake.func @mcAndLSQ
// CHECK: mem_controller[%{{.*}} : memref<16xi32>] start(%{{.*}}) groups(bb0: [load(%{{.*}})], bb1 %{{.*}}: [store(%{{.*}}, %{{.*}}), load(%{{.*}})], bb2 %{{.*}}: []) lsq(%{{.*}}, %{{.*}}, %{{.*}}) end(%{{.*}}) : <i4>, <i32>, <i32>
// CHECK: lsq[MC] groups(%{{.*}}: [load(%{{.*}}), store(%{{.*}}, %{{.*}})]) mc(%{{.*}}) : <i4>, <i32>
handshake.func @mcAndLSQ(%mem: memref<16xi32>, %memStart: !handshake.control<>, %a0: !handshake.channel<i4>, %a1: !handshake.channel<i4>, %a2: !handshake.channel<i4>, %a3: !handshake.channel<i4>, %a4: !handshake.channel<i4>, %d: !handshake.channel<i32>, %d2: !handshake.channel<i32>, %c1: !handshake.channel<i32>, %c2: !handshake.channel<i32>, %g: !handshake.control<>, %ctrlEnd: !handshake.control<>, %start: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mem", "memStart", "a0", "a1", "a2", "a3", "a4", "d", "d2", "c1", "c2", "g", "ctrlEnd", "start"], resNames = ["out0", "out1", "out2", "memEnd", "end"]} {
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
handshake.func @masterLSQ(%mem: memref<8xi16>, %memStart: !handshake.control<>, %a0: !handshake.channel<i3>, %a1: !handshake.channel<i3>, %a2: !handshake.channel<i3>, %d: !handshake.channel<i16>, %g0: !handshake.control<>, %g1: !handshake.control<>, %ctrlEnd: !handshake.control<>, %start: !handshake.control<>, ...) -> (!handshake.channel<i16>, !handshake.channel<i16>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mem", "memStart", "a0", "a1", "a2", "d", "g0", "g1", "ctrlEnd", "start"], resNames = ["out0", "out1", "memEnd", "end"]} {
  %ldData:2, %memEnd = lsq[%mem : memref<8xi16>] start(%memStart) groups(%g0: [store(%stAddr, %stData)], %g1: [load(%ldAddr0), load(%ldAddr1)]) end(%ctrlEnd) : <i3>, <i16>
  %stAddr, %stData = store[%a0] %d {handshake.bb = 0 : ui32} : <i3>, <i16>, <i3>, <i16>
  %ldAddr0, %v0 = load[%a1] %ldData#0 {handshake.bb = 1 : ui32} : <i3>, <i16>, <i3>, <i16>
  %ldAddr1, %v1 = load[%a2] %ldData#1 {handshake.bb = 1 : ui32} : <i3>, <i16>, <i3>, <i16>
  end {handshake.bb = 1 : ui32} %v0, %v1, %memEnd, %start : <i16>, <i16>, <>, <>
}
