// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.json annotate-list=IOGSingleToken,EntryTokenOrder,SingleEntryToken,ExitTokenOrder" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.json" -o /dev/null \
// RUN:   | FileCheck %s
// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.consec.json annotate-list=IOGConsecutiveTokens" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.consec.json" -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=CONSEC

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
  // The pending entry token and a token in another slot of its IOG were copied
  // by the forks between the entry and the next slot. The IOGs are listed in
  // the order in which the search finds them, hence CONSEC-DAG
  // CONSEC-DAG: IOGConsecutiveTokens: ((start_valid & (count(testbench.end_full) > 0)) -> fork0.outs_0_sent)
  // CONSEC-DAG: IOGConsecutiveTokens: ((n_valid & (count(buffer0.outs_valid_i) > 0)) -> fork1.outs_1_sent)
  %startFork:2 = fork [2] %start {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
  // The control merge holds the entry token (index 0), and has not sent its
  // index yet, only if the buffer after it is empty
  // CHECK: EntryTokenOrder: (((control_merge0.slot_full & (control_merge0.data = FALSE) & !control_merge0.index_sent)) -> (! buffer2.outs_valid_i))
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

// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.iog.json annotate-list=IOGSingleToken" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.iog.json" -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=IOG

// x + x: the copies of x on both outputs of the fork meet at the adder before
// any slot, so only the IOG through the first output is kept
handshake.func @addSelf(%x: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
  // IOG-NOT: fork20.outs_1_sent
  // IOG:     IOGSingleToken: (count(x_valid, testbench.end_full) = (1 + count(fork20.outs_0_sent)))
  // IOG-NOT: fork20.outs_1_sent
  %xFork:2 = fork [2] %x {handshake.name = "fork20"} : <i32>
  %sum = addi %xFork#0, %xFork#1 {handshake.name = "addi20"} : <i32>
  end {handshake.name = "end20"} %sum, %start : <i32>, <>
}

// RUN: dynamatic-opt %s \
// RUN:   --handshake-annotate-properties="json-path=%t.exit.json annotate-list=ExitTokenOrder" \
// RUN:   --exp-test-print-formal-properties="json-path=%t.exit.json" -o /dev/null \
// RUN:   | FileCheck %s --check-prefix=EXIT

// A loop whose decider sends its exit decision through a fork to two branches.
// The buffer before the decider keeps the decider in an IOG; without it, the
// copies of the counter would meet at a branch before any slot
handshake.func @exitPaths(%n: <i32>, %start: <>) -> (out0: <i32>, end: <>) {
  %startFork:2 = fork [2] %start {handshake.bb = 0 : ui32, handshake.name = "fork30"} : <>
  %indexBuf = buffer %index, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
  %i = mux %indexBuf [%n, %iBack] {handshake.bb = 1 : ui32, handshake.name = "mux30"} : <i1>, [<i32>, <i32>] to <i32>
  %iFork:2 = fork [2] %i {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <i32>
  %ctrl, %index = control_merge [%startFork#1, %ctrlBack] {handshake.bb = 1 : ui32, handshake.name = "control_merge30"} : [<>, <>] to <>, <i1>
  %zero = source {handshake.bb = 1 : ui32, handshake.name = "source30"} : <>
  %c0 = constant %zero {value = 0 : i32, handshake.bb = 1 : ui32, handshake.name = "constant30"} : <>, <i32>
  %iBuf = buffer %iFork#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i32>
  %cond = cmpi ne, %iBuf, %c0 {handshake.bb = 1 : ui32, handshake.name = "cmpi30"} : <i32>
  // While the exit decision is on a path to a branch, the ancestors of the
  // decider in its IOG are empty: here, for the IOG of %n, the entry, the slot
  // before the decider and the back edge. The slot before the decider only
  // counts until the fork after the decider has sent the decision to that path.
  // There is one invariant per IOG and path, in the order of the walk to the
  // branches, which does not matter, hence EXIT-DAG
  // EXIT-DAG: ExitTokenNoAncestors: ((count((buffer34.outs_valid_i & (buffer34.data = FALSE))) > 0) -> (count((n_valid & !fork31.outs_0_sent), (buffer33.outs_valid_i & !fork32.outs_0_sent), (buffer30.outs_valid_i & !fork31.outs_0_sent)) = 0))
  // EXIT-DAG: ExitTokenNoAncestors: ((count((buffer35.outs_valid_i & (buffer35.data = FALSE))) > 0) -> (count((n_valid & !fork31.outs_0_sent), (buffer33.outs_valid_i & !fork32.outs_1_sent), (buffer30.outs_valid_i & !fork31.outs_0_sent)) = 0))
  %condFork:2 = fork [2] %cond {handshake.bb = 1 : ui32, handshake.name = "fork32"} : <i1>
  %condBuf0 = buffer %condFork#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
  %condBuf1 = buffer %condFork#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer35"} : <i1>
  %iTrue, %iFalse = cond_br %condBuf0, %iFork#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br30"} : <i1>, <i32>
  %ctrlTrue, %ctrlFalse = cond_br %condBuf1, %ctrl {handshake.bb = 1 : ui32, handshake.name = "cond_br31"} : <i1>, <>
  %iBack = buffer %iTrue, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i32>
  %ctrlBack = buffer %ctrlTrue, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1, dvLatency = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <>
  sink %ctrlFalse {handshake.name = "sink30"} : <>
  end {handshake.bb = 2 : ui32, handshake.name = "end30"} %iFalse, %startFork#0 : <i32>, <>
}
