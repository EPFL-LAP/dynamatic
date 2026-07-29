// RUN: dynamatic-opt --lower-handshake-to-hw=instrument-ii=true %s | FileCheck %s
// RUN: dynamatic-opt --lower-handshake-to-hw %s | FileCheck %s --check-prefix=NOINSTR

// Both muxes are fired by the loop header's control merge (their selects are
// the merge's index, through 'fork3'), so both are header muxes of the loop
// and both are observed at their select channels; nothing else is. The merge
// input fed from outside the loop is its first, so an activation's first
// iteration is marked by the select value 0.
// CHECK-LABEL:   hw.module @selfLoop
// CHECK:           %[[SEL1:.*]], %[[SEL0:.*]] = hw.instance "fork3"
// CHECK:           hw.instance "ii_monitor_control_merge0" @ii_monitor_control_merge0(
// CHECK-SAME:        sel0: %[[SEL0]]
// CHECK-SAME:        sel1: %[[SEL1]]
// CHECK-SAME:        clk:
// CHECK-SAME:        rst:
// CHECK-SAME:      ) -> ()
// CHECK:         hw.module.extern @ii_monitor_control_merge0(
// CHECK-SAME:      in %{{[[:alnum:]]+}} : !handshake.channel<i1>
// CHECK-SAME:      in %{{[[:alnum:]]+}} : !handshake.channel<i1>
// CHECK-SAME:      in %{{[[:alnum:]]+}} : i1
// CHECK-SAME:      attributes {hw.name = "ii_monitor", hw.parameters = {INIT_SELECTS = "[0]", LOOP_DEPTH = 1 : ui32, LOOP_MAX_DEPTH = 1 : ui32}}

// Without the option, no monitor is inserted.
// NOINSTR-NOT: ii_monitor

module {
  handshake.func @selfLoop(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["in0", "in1", "start"], resNames = ["end"]} {
    %0:2 = fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1 = mux %5#1 [%arg0, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %2:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %3 = mux %5#0 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %4:2 = fork [2] %3 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%0#1, %trueResult_2]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %5:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %6 = cmpi eq, %2#0, %4#0 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %7:3 = fork [3] %6 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %7#0, %2#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %7#1, %4#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %7#2, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <>
    sink %falseResult_3 {handshake.name = "sink3"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %0#0 : <>
  }
}
