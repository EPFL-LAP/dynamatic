// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=IndexValueRange" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s

// A control merge with three inputs selects two muxes: its index has three
// values in two bits, so each state of the index holds one of the three values
handshake.func @threeWay(%a: <i32>, %b: <i32>, %c: <i32>, %d: <i32>, %e: <i32>, %f: <i32>, %start: <>) -> (out0: <i32>, out1: <i32>, end: <>) {
  %s:4 = fork [4] %start {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
  // CHECK: IndexValueRange: (toint(control_merge0.slot_full) = (toint((control_merge0.slot_full & (control_merge0.data = 0ud2_0))) + toint((control_merge0.slot_full & (control_merge0.data = 0ud2_1))) + toint((control_merge0.slot_full & (control_merge0.data = 0ud2_2)))))
  // CHECK: IndexValueRange: (toint(control_merge0.index_sent) = (toint((control_merge0.index_sent & (control_merge0.ins = 0ud2_0))) + toint((control_merge0.index_sent & (control_merge0.ins = 0ud2_1))) + toint((control_merge0.index_sent & (control_merge0.ins = 0ud2_2)))))
  %ctrl, %index = control_merge [%s#0, %s#1, %s#2] {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>] to <>, <i2>
  // CHECK: IndexValueRange: (toint(buffer0.outs_valid_i) = (toint((buffer0.outs_valid_i & (buffer0.data = 0ud2_0))) + toint((buffer0.outs_valid_i & (buffer0.data = 0ud2_1))) + toint((buffer0.outs_valid_i & (buffer0.data = 0ud2_2)))))
  %indexBuf = buffer %index, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i2>
  // CHECK: IndexValueRange: (toint(fork1.outs_0_sent) = (toint((fork1.outs_0_sent & (fork1.ins = 0ud2_0))) + toint((fork1.outs_0_sent & (fork1.ins = 0ud2_1))) + toint((fork1.outs_0_sent & (fork1.ins = 0ud2_2)))))
  // CHECK: IndexValueRange: (toint(fork1.outs_1_sent) = (toint((fork1.outs_1_sent & (fork1.ins = 0ud2_0))) + toint((fork1.outs_1_sent & (fork1.ins = 0ud2_1))) + toint((fork1.outs_1_sent & (fork1.ins = 0ud2_2)))))
  %indexFork:2 = fork [2] %indexBuf {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i2>
  %x = mux %indexFork#0 [%a, %b, %c] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>] to <i32>
  %y = mux %indexFork#1 [%d, %e, %f] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i32>, <i32>, <i32>] to <i32>
  sink %ctrl {handshake.name = "sink0"} : <>
  end {handshake.bb = 1 : ui32, handshake.name = "end0"} %x, %y, %s#3 : <i32>, <i32>, <>
}
