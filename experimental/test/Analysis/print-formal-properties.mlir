// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=EagerForkNotAllOutputSent" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s

handshake.func @fork(%a: !handshake.channel<i32>, %start: !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["a", "start"], resNames = ["out0", "out1", "end"]} {
  // CHECK: EagerForkNotAllOutputSent: (count(fork0.outs_0_sent, fork0.outs_1_sent) < 2)
  %0:2 = fork [2] %a {handshake.name = "fork0"} : <i32>
  end {handshake.name = "end0"} %0#0, %0#1, %start : <i32>, <i32>, <>
}
