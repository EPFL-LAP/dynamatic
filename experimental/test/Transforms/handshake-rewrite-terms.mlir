// RUN: dynamatic-opt %s --handshake-rewrite-terms --split-input-file | FileCheck %s --implicit-check-not=control_merge

// The pass runs before materialization, so values may have multiple users and
// the circuits below contain no forks or sinks.

// Rewrite E (Fig. 8e): a Branch whose two outputs feed a Mux driven by the same
// condition is a pointless if-then-else.
// CHECK-LABEL: handshake.func @ifThenElseMux(
// CHECK-SAME:    %[[D:arg[0-9]+]]: !handshake.channel<i32>
// CHECK-NOT:     cond_br
// CHECK-NOT:     mux
// CHECK:         end {{.*}}%[[D]] : <i32>
handshake.func @ifThenElseMux(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %true, %false = cond_br %c, %d : <i1>, <i32>
  %res = mux %c [%false, %true] : <i1>, [<i32>, <i32>] to <i32>
  end %res : <i32>
}

// -----

// Same as above, with a Merge instead of a Mux.
// CHECK-LABEL: handshake.func @ifThenElseMerge(
// CHECK-SAME:    %[[D:arg[0-9]+]]: !handshake.channel<i32>
// CHECK-NOT:     cond_br
// CHECK-NOT:     merge
// CHECK:         end {{.*}}%[[D]] : <i32>
handshake.func @ifThenElseMerge(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %true, %false = cond_br %c, %d : <i1>, <i32>
  %res = merge %true, %false : <i32>
  end %res : <i32>
}

// -----

// The Mux selects each token from the side of the Branch that does not
// produce it: this is not an if-then-else and must not be simplified.
// CHECK-LABEL: handshake.func @ifThenElseMuxSwappedInputs(
// CHECK:         %[[MUX:[0-9]+]] = mux
// CHECK:         end {{.*}}%[[MUX]] : <i32>
handshake.func @ifThenElseMuxSwappedInputs(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %true, %false = cond_br %c, %d : <i1>, <i32>
  %res = mux %c [%true, %false] : <i1>, [<i32>, <i32>] to <i32>
  end %res : <i32>
}

// -----

// The Mux select is unrelated to the Branch condition: must not be simplified.
// CHECK-LABEL: handshake.func @ifThenElseMuxOtherSelect(
// CHECK:         %[[MUX:[0-9]+]] = mux
// CHECK:         end {{.*}}%[[MUX]] : <i32>
handshake.func @ifThenElseMuxOtherSelect(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, %s: !handshake.channel<i1>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %true, %false = cond_br %c, %d : <i1>, <i32>
  %res = mux %s [%false, %true] : <i1>, [<i32>, <i32>] to <i32>
  end %res : <i32>
}

// -----

// Two identical Suppresses feeding a Merge produce two tokens for each token
// of %d when %c is false: must not be simplified.
// CHECK-LABEL: handshake.func @identicalSuppressesMerge(
// CHECK:         %[[MERGE:[0-9]+]] = merge
// CHECK:         end {{.*}}%[[MERGE]] : <i32>
handshake.func @identicalSuppressesMerge(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %true0, %false0 = cond_br %c, %d : <i1>, <i32>
  %true1, %false1 = cond_br %c, %d : <i1>, <i32>
  %res = merge %false0, %false1 : <i32>
  end %res : <i32>
}

// -----

// Rewrite B (Fig. 8b): the index of a loop-header Control Merge is replaced by
// an Init of the loop condition. The input coming from outside the loop is the
// first one, so the Init produces 0 at the first iteration.
// CHECK-LABEL: handshake.func @loopInitialToken0(
// CHECK-SAME:    %[[X0:arg[0-9]+]]: !handshake.channel<i32>
// CHECK:         %[[INIT:[0-9]+]] = init %[[COND:[0-9]+]] {{.*}} : <i1>
// CHECK:         mux %[[INIT]] [%[[X0]], %{{.*}}] {{.*}} : <i1>, [<i32>, <i32>] to <i32>
// CHECK:         %[[COND]] = cmpi ult
handshake.func @loopInitialToken0(%x0: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %ctrl, %idx = control_merge [%start, %ctrlBack] : [<>, <>] to <>, <i1>
  %x = mux %idx [%x0, %xBack] : <i1>, [<i32>, <i32>] to <i32>
  %one = constant %ctrl {value = 1 : i32} : <>, <i32>
  %bound = constant %ctrl {value = 10 : i32} : <>, <i32>
  %xNext = addi %x, %one : <i32>
  %c = cmpi ult, %xNext, %bound : <i32>
  %xBack, %xExit = cond_br %c, %xNext : <i1>, <i32>
  %ctrlBack, %ctrlExit = cond_br %c, %ctrl : <i1>, <>
  end %xExit : <i32>
}

// -----

// Same as above, with the input coming from outside the loop second: the Init
// must produce 1 at the first iteration.
// CHECK-LABEL: handshake.func @loopInitialToken1(
// CHECK-SAME:    %[[X0:arg[0-9]+]]: !handshake.channel<i32>
// CHECK:         %[[INIT:[0-9]+]] = init %[[COND:[0-9]+]] {{.*}} : <i1>
// CHECK:         %[[SEL:[0-9]+]] = noti %[[INIT]]
// CHECK:         mux %[[SEL]] [%{{.*}}, %[[X0]]] {{.*}} : <i1>, [<i32>, <i32>] to <i32>
// CHECK:         %[[COND]] = cmpi ult
handshake.func @loopInitialToken1(%x0: !handshake.channel<i32>, %start: !handshake.control<>) -> !handshake.channel<i32> {
  %ctrl, %idx = control_merge [%ctrlBack, %start] : [<>, <>] to <>, <i1>
  %x = mux %idx [%xBack, %x0] : <i1>, [<i32>, <i32>] to <i32>
  %one = constant %ctrl {value = 1 : i32} : <>, <i32>
  %bound = constant %ctrl {value = 10 : i32} : <>, <i32>
  %xNext = addi %x, %one : <i32>
  %c = cmpi ult, %xNext, %bound : <i32>
  %xBack, %xExit = cond_br %c, %xNext : <i1>, <i32>
  %ctrlBack, %ctrlExit = cond_br %c, %ctrl : <i1>, <>
  end %xExit : <i32>
}
