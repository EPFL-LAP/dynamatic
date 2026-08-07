// RUN: dynamatic-opt --lower-handshake-to-hw=instrument-ii=true %s | FileCheck %s

// In a nest, each loop observes exactly its own header muxes: they are found
// by walking the fan-out of each loop's own control merge index, so
// 'muxOuter' (fired by the outer merge) belongs to the outer loop's monitor
// and 'muxInner' (fired by the inner merge) to the inner one's, without ever
// classifying where the muxes' data operands come from. Both selects here
// feed their mux directly from the merge, so the observed channel is the
// index output itself.

// CHECK-LABEL:   hw.module @nestedLoops
// CHECK:           %{{.*}}, %[[SEL_OUTER:.*]] = hw.instance "control_merge0"
// CHECK:           %{{.*}}, %[[SEL_INNER:.*]] = hw.instance "control_merge1"

// The outer loop observes only 'muxOuter's select.
// CHECK:           hw.instance "ii_monitor_control_merge0" @ii_monitor_control_merge0(
// CHECK-SAME:        sel0: %[[SEL_OUTER]]
// CHECK-SAME:      ) -> ()

// The inner loop observes only 'muxInner's select.
// CHECK:           hw.instance "ii_monitor_control_merge1" @ii_monitor_control_merge1(
// CHECK-SAME:        sel0: %[[SEL_INNER]]
// CHECK-SAME:      ) -> ()

// CHECK:         hw.module.extern @ii_monitor_control_merge0(
// CHECK-SAME:      in %{{[[:alnum:]]+}} : !handshake.channel<i1>
// CHECK-SAME:      in %{{[[:alnum:]]+}} : i1
// CHECK-SAME:      attributes {hw.name = "ii_monitor", hw.parameters = {INIT_SELECTS = "[0]", LOOP_DEPTH = 1 : ui32, LOOP_MAX_DEPTH = 2 : ui32}}

// CHECK:         hw.module.extern @ii_monitor_control_merge1(
// CHECK-SAME:      attributes {hw.name = "ii_monitor", hw.parameters = {INIT_SELECTS = "[0]", LOOP_DEPTH = 2 : ui32, LOOP_MAX_DEPTH = 2 : ui32}}

module {
  handshake.func @nestedLoops(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, ...) -> !handshake.control<> attributes {argNames = ["in0", "in1", "start"], resNames = ["end"]} {
    %start:2 = fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %bound:2 = fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>

    // Outer loop header.
    %outerVal = mux %outerIndex [%arg0, %outerBackVal] {handshake.bb = 1 : ui32, handshake.name = "muxOuter"} : <i1>, [<i32>, <i32>] to <i32>
    %outerVals:2 = fork [2] %outerVal {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %outerCtrl, %outerIndex = control_merge [%start#1, %outerBackCtrl]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>

    // Inner loop header.
    %innerVal = mux %innerIndex [%outerVals#0, %innerBackVal] {handshake.bb = 2 : ui32, handshake.name = "muxInner"} : <i1>, [<i32>, <i32>] to <i32>
    %innerVals:3 = fork [3] %innerVal {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %innerCtrl, %innerIndex = control_merge [%outerCtrl, %innerBackCtrl]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %innerCond = cmpi eq, %innerVals#0, %bound#0 {handshake.bb = 2 : ui32, handshake.name = "cmpiInner"} : <i32>
    %innerConds:2 = fork [2] %innerCond {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %innerBackVal, %innerExitVal = cond_br %innerConds#0, %innerVals#1 {handshake.bb = 2 : ui32, handshake.name = "cond_brInnerVal"} : <i1>, <i32>
    %innerBackCtrl, %innerExitCtrl = cond_br %innerConds#1, %innerCtrl {handshake.bb = 2 : ui32, handshake.name = "cond_brInnerCtrl"} : <i1>, <>
    sink %innerVals#2 {handshake.name = "sink0"} : <i32>

    // Outer loop latch.
    %outerCond = cmpi eq, %innerExitVal, %bound#1 {handshake.bb = 3 : ui32, handshake.name = "cmpiOuter"} : <i32>
    %outerConds:2 = fork [2] %outerCond {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <i1>
    %outerBackVal, %outerExitVal = cond_br %outerConds#0, %outerVals#1 {handshake.bb = 3 : ui32, handshake.name = "cond_brOuterVal"} : <i1>, <i32>
    %outerBackCtrl, %outerExitCtrl = cond_br %outerConds#1, %innerExitCtrl {handshake.bb = 3 : ui32, handshake.name = "cond_brOuterCtrl"} : <i1>, <>
    sink %outerExitVal {handshake.name = "sink1"} : <i32>
    sink %outerExitCtrl {handshake.name = "sink2"} : <>

    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %start#0 : <>
  }
}
