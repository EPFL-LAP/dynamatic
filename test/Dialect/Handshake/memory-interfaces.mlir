// RUN: dynamatic-opt %s --split-input-file | dynamatic-opt --split-input-file | FileCheck %s

// CHECK-LABEL: handshake.func @mcOnly(
// CHECK:         %[[LD:.*]]:2, %[[END:.*]] = mem_controller[%{{[^,) ]+}} : memref<64xi32>] start(%{{[^,) ]+}}) ctrl(%{{[^,) ]+}}) ldAddr(%{{[^,) ]+}}, %{{[^,) ]+}}) stAddr(%{{[^,) ]+}}) stData(%{{[^,) ]+}}) end(%{{[^,) ]+}})
// CHECK-SAME:      {groups = [#handshake.mem_group<0, [ld]>, #handshake.mem_group<1, ctrl, [ld, st]>]}
// CHECK-SAME:      : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
handshake.func @mcOnly(%mem: memref<64xi32>, %memStart: !handshake.control<>, %ctrl: !handshake.channel<i32>, %addr0: !handshake.channel<i32>, %addr1: !handshake.channel<i32>, %stAddr: !handshake.channel<i32>, %stData: !handshake.channel<i32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) {
  %ldData:2, %memEnd = mem_controller [%mem : memref<64xi32>] start(%memStart) ctrl(%ctrl) ldAddr(%addr0, %addr1) stAddr(%stAddr) stData(%stData) end(%start) {groups = [#handshake.mem_group<0, [ld]>, #handshake.mem_group<1, ctrl, [ld, st]>]} : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
  end %ldData#0, %ldData#1, %memEnd : <i32>, <i32>, <>
}

// -----

// CHECK-LABEL: handshake.func @lsqMaster(
// CHECK:         %{{[^,) ]+}}, %{{[^,) ]+}} = lsq[%{{[^,) ]+}} : memref<64xi32>] start(%{{[^,) ]+}}) ctrl(%{{[^,) ]+}}, %{{[^,) ]+}}) ldAddr(%{{[^,) ]+}}) stAddr(%{{[^,) ]+}}) stData(%{{[^,) ]+}}) end(%{{[^,) ]+}})
// CHECK-SAME:      {groups = [#handshake.mem_group<0, ctrl, [ld]>, #handshake.mem_group<1, ctrl, [st]>]}
// CHECK-SAME:      : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
handshake.func @lsqMaster(%mem: memref<64xi32>, %memStart: !handshake.control<>, %ctrl0: !handshake.control<>, %ctrl1: !handshake.control<>, %ldAddr: !handshake.channel<i32>, %stAddr: !handshake.channel<i32>, %stData: !handshake.channel<i32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>) {
  %ldData, %memEnd = lsq [%mem : memref<64xi32>] start(%memStart) ctrl(%ctrl0, %ctrl1) ldAddr(%ldAddr) stAddr(%stAddr) stData(%stData) end(%start) {groups = [#handshake.mem_group<0, ctrl, [ld]>, #handshake.mem_group<1, ctrl, [st]>]} : (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
  end %ldData, %memEnd : <i32>, <>
}

// -----

// The MC has a control-only group (block 2) for the stores that go through
// the LSQ.

// CHECK-LABEL: handshake.func @mcAndLSQ(
// CHECK:         %[[MC_LD:.*]], %[[LSQ_LD:.*]], %{{[^,) ]+}} = mem_controller[%{{[^,) ]+}} : memref<64xi32>] start(%{{[^,) ]+}}) ctrl(%{{[^,) ]+}}, %{{[^,) ]+}}) ldAddr(%{{[^,) ]+}}) fromLSQ(%[[LD_ADDR:.*]], %[[ST_ADDR:.*]], %[[ST_DATA:.*]]) end(%{{[^,) ]+}})
// CHECK-SAME:      {groups = [#handshake.mem_group<1, ctrl, [ld]>, #handshake.mem_group<2, ctrl, []>]}
// CHECK:         %{{[^,) ]+}}, %[[LD_ADDR]], %[[ST_ADDR]], %[[ST_DATA]] = lsq[MC] ctrl(%{{[^,) ]+}}) ldAddr(%{{[^,) ]+}}) stAddr(%{{[^,) ]+}}) stData(%{{[^,) ]+}}) fromMC(%[[LSQ_LD]])
// CHECK-SAME:      {groups = [#handshake.mem_group<0, ctrl, [ld, st]>]}
// CHECK-SAME:      : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
handshake.func @mcAndLSQ(%mem: memref<64xi32>, %memStart: !handshake.control<>, %mcCtrl1: !handshake.channel<i32>, %mcCtrl2: !handshake.channel<i32>, %lsqCtrl: !handshake.control<>, %mcLdAddr: !handshake.channel<i32>, %ldAddr: !handshake.channel<i32>, %stAddr: !handshake.channel<i32>, %stData: !handshake.channel<i32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) {
  %mcLdData, %ldDataToLSQ, %memEnd = mem_controller [%mem : memref<64xi32>] start(%memStart) ctrl(%mcCtrl1, %mcCtrl2) ldAddr(%mcLdAddr) fromLSQ(%ldAddrToMC, %stAddrToMC, %stDataToMC) end(%start) {groups = [#handshake.mem_group<1, ctrl, [ld]>, #handshake.mem_group<2, ctrl, []>]} : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
  %lsqLdData, %ldAddrToMC, %stAddrToMC, %stDataToMC = lsq [MC] ctrl(%lsqCtrl) ldAddr(%ldAddr) stAddr(%stAddr) stData(%stData) fromMC(%ldDataToLSQ) {groups = [#handshake.mem_group<0, ctrl, [ld, st]>]} : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
  end %mcLdData, %lsqLdData, %memEnd : <i32>, <i32>, <>
}
