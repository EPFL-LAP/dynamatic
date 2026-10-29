// RUN: dynamatic-opt %s --handshake-annotate-properties="json-path=%t.json annotate-list=AbsenceOfBackpressure,EagerForkNotAllOutputSent" > /dev/null
// RUN: FileCheck %s < %t.json

// The properties to check come first, in the order of the channels, followed
// by the invariants

handshake.func @forkBuffer(%a: !handshake.channel<i32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["a", "start"], resNames = ["out0", "end"]} {
  // Absence of backpressure on the first output of the fork, whose user is the
  // buffer:
  // fork0.outs_0_valid -> buffer0.ins_ready
  // CHECK:      "check": null,
  // CHECK-NEXT: "expr": {
  // CHECK-NEXT:   "op": "implies",
  // CHECK-NEXT:   "operands": [
  // CHECK-NEXT:     {
  // CHECK-NEXT:       "index": 0,
  // CHECK-NEXT:       "op": "signal",
  // CHECK-NEXT:       "port": "outs_0",
  // CHECK-NEXT:       "side": "result",
  // CHECK-NEXT:       "signal": "valid",
  // CHECK-NEXT:       "unit": "fork0"
  // CHECK-NEXT:     },
  // CHECK-NEXT:     {
  // CHECK-NEXT:       "index": 0,
  // CHECK-NEXT:       "op": "signal",
  // CHECK-NEXT:       "port": "ins",
  // CHECK-NEXT:       "side": "operand",
  // CHECK-NEXT:       "signal": "ready",
  // CHECK-NEXT:       "unit": "buffer0"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   ]
  // CHECK-NEXT: },
  // CHECK-NEXT: "id": 0,
  // CHECK-NEXT: "kind": "AbsenceOfBackpressure",
  // CHECK-NEXT: "tag": "OPT"
  //
  // Invariant: not all outputs of the fork are in the sent state at the same
  // time, i.e., count(fork0.outs_0_sent, fork0.outs_1_sent) < 2:
  // CHECK:        "op": "lt",
  // CHECK-NEXT:   "operands": [
  // CHECK-NEXT:     {
  // CHECK-NEXT:       "op": "count",
  // CHECK-NEXT:       "operands": [
  // CHECK-NEXT:         {
  // CHECK-NEXT:           "op": "state",
  // CHECK-NEXT:           "state": {
  // CHECK-NEXT:             "inner": {
  // CHECK-NEXT:               "channel_name": "outs_0",
  // CHECK-NEXT:               "channel_size": 32,
  // CHECK-NEXT:               "operation": "fork0"
  // CHECK-NEXT:             },
  // CHECK-NEXT:             "type": "EagerForkSent"
  // CHECK-NEXT:           }
  // CHECK-NEXT:         },
  // CHECK-NEXT:         {
  // CHECK-NEXT:           "op": "state",
  // CHECK-NEXT:           "state": {
  // CHECK-NEXT:             "inner": {
  // CHECK-NEXT:               "channel_name": "outs_1",
  // CHECK-NEXT:               "channel_size": 32,
  // CHECK-NEXT:               "operation": "fork0"
  // CHECK-NEXT:             },
  // CHECK-NEXT:             "type": "EagerForkSent"
  // CHECK-NEXT:           }
  // CHECK-NEXT:         }
  // CHECK-NEXT:       ]
  // CHECK-NEXT:     },
  // CHECK-NEXT:     {
  // CHECK-NEXT:       "op": "int",
  // CHECK-NEXT:       "value": 2
  // CHECK-NEXT:     }
  // CHECK-NEXT:   ]
  // CHECK-NEXT: },
  // CHECK-NEXT: "id": 3,
  // CHECK-NEXT: "kind": "EagerForkNotAllOutputSent",
  // CHECK-NEXT: "tag": "INVAR"
  %0:2 = fork [2] %a {handshake.name = "fork0"} : <i32>
  %1 = buffer %0#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.name = "buffer0"} : <i32>
  %2 = addi %1, %0#1 {handshake.name = "addi0"} : <i32>
  end {handshake.name = "end0"} %2, %start : <i32>, <>
}
