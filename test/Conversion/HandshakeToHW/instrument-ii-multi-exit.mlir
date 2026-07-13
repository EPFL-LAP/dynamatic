// RUN: dynamatic-opt --lower-handshake-to-hw=instrument-ii=true %s | FileCheck %s

// CHECK-LABEL:   hw.module @multiExit
// CHECK:         %{{.*}}, %[[ENTRY:.*]] = hw.instance "fork0"
// CHECK:         %{{.*}}, %[[EXIT_A:.*]] = hw.instance "cond_brExtra"

// Each exit channel is forked: the original consumer keeps the first output, the
// monitor's merge observes the second.
// CHECK:         %[[FORK_A_ORIG:.*]], %[[FORK_A_TAP:.*]] = hw.instance "ii_exit_fork" {{.*}}(ins: %[[EXIT_A]]
// CHECK:         hw.instance "sinkCE" {{.*}}(ins: %[[FORK_A_ORIG]]
// CHECK:         %[[BACKEDGE:.*]], %[[EXIT_B:.*]] = hw.instance "cond_br2"
// CHECK:         %[[FORK_B_ORIG:.*]], %[[FORK_B_TAP:.*]] = hw.instance "ii_exit_fork0" {{.*}}(ins: %[[EXIT_B]]
// CHECK:         hw.instance "sink3" {{.*}}(ins: %[[FORK_B_ORIG]]

// The two tapped copies are merged and the merge's outputs are consumed by sinks.
// CHECK:         %[[MERGE_OUT:.*]], %[[MERGE_IDX:.*]] = hw.instance "ii_exit_merge" @handshake_control_merge_{{[0-9]+}}(
// CHECK-SAME:      ins_0: %[[FORK_A_TAP]]
// CHECK-SAME:      ins_1: %[[FORK_B_TAP]]
// CHECK:         hw.instance "ii_exit_sink{{[0-9]*}}" {{.*}}(ins: %[[MERGE_OUT]]
// CHECK:         hw.instance "ii_exit_sink{{[0-9]*}}" {{.*}}(ins: %[[MERGE_IDX]]

// The monitor observes the control merge's entry and back-edge inputs and the
// merged exit channel.
// CHECK:         hw.instance "ii_monitor_control_merge0" @ii_monitor_1_1(
// CHECK-SAME:      entry: %[[ENTRY]]
// CHECK-SAME:      backedge: %[[BACKEDGE]]
// CHECK-SAME:      exit: %[[MERGE_OUT]]

module {
  handshake.func @multiExit(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["in0", "in1", "start"], resNames = ["end"]} {
    %0:2 = fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1 = mux %5#1 [%arg0, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %2:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %3 = mux %5#0 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %4:2 = fork [2] %3 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%0#1, %trueResult_2]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %5:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %6 = cmpi eq, %2#0, %4#0 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %7:4 = fork [4] %6 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %7#0, %2#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %7#1, %4#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %ceTrue, %ceFalse = cond_br %7#3, %result {handshake.bb = 1 : ui32, handshake.name = "cond_brExtra"} : <i1>, <>
    sink %ceFalse {handshake.name = "sinkCE"} : <>
    %trueResult_2, %falseResult_3 = cond_br %7#2, %ceTrue {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <>
    sink %falseResult_3 {handshake.name = "sink3"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %0#0 : <>
  }
}
