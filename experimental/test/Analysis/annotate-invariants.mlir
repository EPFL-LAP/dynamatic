// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=IOGSingleToken,EntryTokenOrder,SingleEntryToken,ExitTokenOrder" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s

// A loop that counts down from n, with a buffer on the index of its control
// merge and two buffers on its condition

// One IOGSingleToken invariant per IOG, with its slots and the fork outputs that
// copy its token. The IOGs of %n go either through the data or through the
// condition of the loop's branch; those of %start through the loop's control or
// to the end. The IOGs are listed in the order in which the search finds them,
// which does not matter, hence CHECK-DAG
// CHECK-DAG: IOGSingleToken: (count(n_valid, testbench.end_full, buffer0.outs_valid_i) = (1 + count(fork1.outs_1_sent)))
// CHECK-DAG: IOGSingleToken: (count(n_valid, buffer3.outs_valid_i, buffer4.outs_valid_i, testbench.end_full, buffer0.outs_valid_i) = (1 + count(fork1.outs_0_sent, fork2.outs_0_sent)))
// CHECK-DAG: IOGSingleToken: (count(start_valid, control_merge0.slot_full, buffer2.outs_valid_i, buffer3.outs_valid_i, buffer4.outs_valid_i, buffer1.outs_valid_i, formal.dead_buffer0.full) = (1 + count(fork0.outs_1_sent, control_merge0.index_sent, fork1.outs_0_sent, fork2.outs_1_sent)))
// CHECK-DAG: IOGSingleToken: (count(start_valid, control_merge0.slot_full, buffer1.outs_valid_i, formal.dead_buffer0.full) = (1 + count(fork0.outs_1_sent, control_merge0.outs_sent)))
// CHECK-DAG: IOGSingleToken: (count(start_valid, testbench.end_full) = (1 + count(fork0.outs_0_sent)))
handshake.func @countLoop(%n: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
  %startFork:2 = fork [2] %start {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
  // The control merge holds the entry token (index 0) only if the buffer after
  // it is empty
  // CHECK: EntryTokenOrder: ((control_merge0.slot_full & (control_merge0.data = FALSE)) -> (! buffer2.outs_valid_i))
  %indexBuf = buffer %index, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i1>
  %i = mux %indexBuf [%n, %iBack] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
  %iFork:2 = fork [2] %i {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
  // No token enters the loop while it holds one
  // CHECK: SingleEntryToken: ((count(control_merge0.slot_full, buffer2.outs_valid_i) > 0) -> (count((start_valid & !fork0.outs_1_sent)) = 0))
  %ctrl, %index = control_merge [%startFork#1, %ctrlBack] {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
  %zero = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
  %c0 = constant %zero {value = 0 : i32, handshake.bb = 1 : ui32, handshake.name = "constant0"} : <>, <i32>
  %cond = cmpi ne, %iFork#0, %c0 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
  // The second buffer holds the exit decision (FALSE) only if the first one is
  // empty, unless the branch it goes to has already received it. There is one
  // invariant per branch, in the order of the walk to the branches, which does
  // not matter, hence CHECK-DAG
  // CHECK-DAG: ExitTokenOrder: (((buffer4.outs_valid_i & (buffer4.data = FALSE) & !fork2.outs_1_sent)) -> (! buffer3.outs_valid_i))
  // CHECK-DAG: ExitTokenOrder: (((buffer4.outs_valid_i & (buffer4.data = FALSE) & !fork2.outs_0_sent)) -> (! buffer3.outs_valid_i))
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

// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.flow.json annotate-list=ReconvergentPathFlow" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.flow.json" -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=FLOW

handshake.func @store(%mem: memref<100xi32>, %memStart: <>, %start: <>) -> (mem_end: <>, end: <>) {
  %ctrl:5 = fork [5] %start {handshake.bb = 0 : ui32, handshake.name = "fork10"} : <>
  %count = constant %ctrl#0 {value = 1 : i32, handshake.bb = 0 : ui32, handshake.name = "constant10"} : <>, <i32>
  %addr = constant %ctrl#1 {value = 0 : i7, handshake.bb = 0 : ui32, handshake.name = "constant11"} : <>, <i7>
  %data = constant %ctrl#2 {value = 5 : i32, handshake.bb = 0 : ui32, handshake.name = "constant12"} : <>, <i32>
  // The memory controller takes the address and the data of the store
  // together, so the fork sends the copies to both of them together
  // FLOW: ReconvergentPathFlow: (((toint(fork10.outs_1_sent) * 1) + (toint(fork10.outs_2_sent) * -1)) = 0)
  %stAddr, %stData = store[%addr] %data {handshake.bb = 0 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store10"} : <i7>, <i32>, <i7>, <i32>
  %memEnd = mem_controller[%mem : memref<100xi32>] start(%memStart) groups(bb0 %count: [store(%stAddr, %stData)]) end(%ctrl#3) {handshake.name = "mem_controller10"} : <i7>, <i32>, <i32>
  end {handshake.bb = 0 : ui32, handshake.name = "end10"} %memEnd, %ctrl#4 : <>, <>
}
