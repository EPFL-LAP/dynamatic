// RUN: dynamatic-opt %s --split-input-file \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=ReconvergentPathFlow" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s

// A store to memory: the memory controller holds the token of %memStart while
// it runs, until the end of the function
handshake.func @store(%mem: memref<100xi32>, %memStart: <>, %start: <>) -> (mem_end: <>, end: <>) {
  %ctrl:5 = fork [5] %start {handshake.bb = 0 : ui32, handshake.name = "fork10"} : <>
  %count = constant %ctrl#0 {value = 1 : i32, handshake.bb = 0 : ui32, handshake.name = "constant10"} : <>, <i32>
  %addr = constant %ctrl#1 {value = 0 : i7, handshake.bb = 0 : ui32, handshake.name = "constant11"} : <>, <i7>
  %data = constant %ctrl#2 {value = 5 : i32, handshake.bb = 0 : ui32, handshake.name = "constant12"} : <>, <i32>
  %stAddr, %stData = store[%addr] %data {handshake.bb = 0 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store10"} : <i7>, <i32>, <i7>, <i32>
  // The token of %memStart is in the argument or in the controller until the
  // controller has received its end control from %start and has stopped
  // CHECK: ReconvergentPathFlow: (((toint(memStart_valid) * 1) + (toint(start_valid) * -1) + (toint(fork10.outs_3_sent) * 1) + (toint(mem_controller10.inner_mc_control.fsm_running) * 1) + (toint(mem_controller10.inner_mc_control.no_more_request) * -1)) = 0)
  // The controller holds the token of its end control once the circuit makes
  // no more requests
  // CHECK: ReconvergentPathFlow: (((toint(fork10.outs_3_sent) * 1) + (toint(fork10.outs_4_sent) * -1) + (toint(mem_controller10.inner_mc_control.no_more_request) * -1)) = 0)
  %memEnd = mem_controller[%mem : memref<100xi32>] start(%memStart) groups(bb0 %count: [store(%stAddr, %stData)]) end(%ctrl#3) {handshake.name = "mem_controller10"} : <i7>, <i32>, <i32>
  end {handshake.bb = 0 : ui32, handshake.name = "end10"} %memEnd, %ctrl#4 : <>, <>
}

// -----

// A loop with a buffer on the index of its control merge: the tokens of each
// index value are related separately
handshake.func @countLoop(%n: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
  %startFork:2 = fork [2] %start {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
  // The tokens of value TRUE (back edge) pass the buffer on their own
  // CHECK: ReconvergentPathFlow: (((toint((buffer2.outs_valid_i & (buffer2.data = TRUE))) * -1) + (toint((control_merge0.slot_full & (control_merge0.data = TRUE))) * -1) + (toint((control_merge0.index_sent & (control_merge0.ins = TRUE))) * 1) + (toint((fork2.outs_0_sent & (fork2.ins = TRUE))) * -1) + (toint((fork2.outs_1_sent & (fork2.ins = TRUE))) * 1) + (toint(buffer0.outs_valid_i) * 1) + (toint(buffer1.outs_valid_i) * -1)) = 0)
  %indexBuf = buffer %index, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i1>
  %i = mux %indexBuf [%n, %iBack] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
  %iFork:2 = fork [2] %i {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
  %ctrl, %index = control_merge [%startFork#1, %ctrlBack] {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
  %zero = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
  %c0 = constant %zero {value = 0 : i32, handshake.bb = 1 : ui32, handshake.name = "constant0"} : <>, <i32>
  %cond = cmpi ne, %iFork#0, %c0 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
  %condBuf0 = buffer %cond, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i1>
  %condBuf1 = buffer %condBuf0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i1>
  %condFork:2 = fork [2] %condBuf1 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
  %iTrue, %iFalse = cond_br %condFork#0, %iFork#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i32>
  %ctrlTrue, %ctrlFalse = cond_br %condFork#1, %ctrl {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <>
  %iBack = buffer %iTrue, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
  %ctrlBack = buffer %ctrlTrue, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
  sink %ctrlFalse {handshake.name = "sink0"} : <>
  end {handshake.bb = 2 : ui32, handshake.name = "end0"} %iFalse, %startFork#0 : <i32>, <>
}
