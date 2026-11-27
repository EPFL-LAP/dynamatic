// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=ReconvergentPathFlow" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s

// A select can produce its result before the token of the data input it does
// not select arrives, and kills that token when it arrives
handshake.func @select(%cond: <i1>, %a: <i32>, %b: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
  %a1 = buffer %a, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 0 : ui32, handshake.name = "buffer0"} : <i32>
  // The token of %a is in its argument or in buffer0 while the condition is
  // in its argument, or the select has fired and owes it a kill
  // CHECK: ReconvergentPathFlow: (((toint(cond_valid) * -1) + (toint(a_valid) * 1) + (toint(buffer0.outs_valid_i) * 1) + (toint((select0.killTrueOwed ? select0.cnt : 0)) * -1)) = 0)
  // The same for %b, related to %start through the end
  // CHECK: ReconvergentPathFlow: (((toint(b_valid) * 1) + (toint(start_valid) * -1) + (toint((select0.killFalseOwed ? select0.cnt : 0)) * -1)) = 0)
  %res = select %cond [%a1, %b] {handshake.bb = 0 : ui32, handshake.name = "select0"} : <i1>, <i32>
  end {handshake.bb = 0 : ui32, handshake.name = "end0"} %res, %start : <i32>, <>
}

