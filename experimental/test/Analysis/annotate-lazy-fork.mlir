// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=ReconvergentPathFlow,IOGSingleToken" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s

// A lazy fork sends its copies in the cycle in which it takes the token, so
// it holds no part of the token: its two copies are in the buffers in equal
// numbers, and each IOG through it counts only the slots
handshake.func @lazyCopies(%start: <>) -> (out0: <>, end: <>) {
  // CHECK: ReconvergentPathFlow: (((toint(start_valid) * -1) + 1 + (toint(buffer0.outs_valid_i) * -1) + (toint(testbench.end_full) * -1)) = 0)
  // CHECK: ReconvergentPathFlow: (((toint(buffer0.outs_valid_i) * -1) + (toint(buffer1.outs_valid_i) * 1)) = 0)
  // CHECK-DAG: IOGSingleToken: (count(start_valid, buffer0.outs_valid_i, testbench.end_full) = 1)
  // CHECK-DAG: IOGSingleToken: (count(start_valid, buffer1.outs_valid_i, testbench.end_full) = 1)
  %copies:2 = lazy_fork [2] %start {handshake.bb = 0 : ui32, handshake.name = "lazy_fork0"} : <>
  %copy0 = buffer %copies#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 0 : ui32, handshake.name = "buffer0"} : <>
  %copy1 = buffer %copies#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 0 : ui32, handshake.name = "buffer1"} : <>
  end {handshake.bb = 0 : ui32, handshake.name = "end0"} %copy0, %copy1 : <>, <>
}
