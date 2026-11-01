// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=CopiedSlotsOfActiveForksAreFull" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s

handshake.func @zeroLatency(%a: !handshake.channel<f32>, %b: !handshake.channel<f32>, %start: !handshake.control<>) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.control<>) attributes {argNames = ["a", "b", "start"], resNames = ["out0", "out1", "end"]} {
  %0 = buffer %a, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.name = "buffer0"} : <f32>
  %1 = cmpf ult, %0, %b {handshake.name = "cmpf0", internal_delay = "0_000000", latency = 0 : i64} : <f32>
  // The comparison has no latency, so the slot the fork copies from is the
  // buffer before it
  // CHECK: CopiedSlotsOfActiveForksAreFull: ((fork0.outs_0_sent | fork0.outs_1_sent) -> buffer0.outs_valid_i)
  %2:2 = fork [2] %1 {handshake.name = "fork0"} : <i1>
  end {handshake.name = "end0"} %2#0, %2#1, %start : <i1>, <i1>, <>
}
