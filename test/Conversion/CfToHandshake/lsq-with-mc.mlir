// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names --split-input-file | FileCheck %s

// An if-else with an LSQ store in the true branch,
// and an MC store and an LSQ store in the false branch
// CHECK-LABEL: handshake.func @lsq_with_mc
func.func @lsq_with_mc(%arg0: i1 {handshake.arg_name = "in0"},
                       %arg1: memref<4xi32> {handshake.arg_name = "mem0"}) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %i0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  memref.store %c42, %arg1[%i0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<4xi32>
  cf.br ^bb3
^bb2:
  %i1 = arith.constant 1 : index
  %i3 = arith.constant 3 : index
  %c7 = arith.constant 7 : i32
  %c9 = arith.constant 9 : i32
  memref.store %c7, %arg1[%i1] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  memref.store %c9, %arg1[%i3] {handshake.name = "store2", handshake.mem_interface = #handshake.mem_interface<LSQ: 1>} : memref<4xi32>
  cf.br ^bb3
^bb3:
  // the start of mem0 produces an allocation token for the LSQ
  // and one for the MC, each branched on in0 to bb1 or bb2.
  // the two branches look the same, so the LSQ is checked first,
  // and the MC then gets the other branch

  // the LSQ is a non-take-end LSQ (an LSQ connected to an MC),
  // so its groups are allocated directly by the outputs of its branch
  // to bb1 and to bb2, with no lazy forks, and it takes no end input
  // CHECK-DAG: lsq[MC] groups(%[[LSQ_MEM0_START_TO_BB1:[^:]+]]: {{.*}}, %[[LSQ_MEM0_START_TO_BB2:[^:]+]]: {{.*}}) mc(
  // CHECK-DAG: %[[LSQ_MEM0_START_TO_BB1]], %[[LSQ_MEM0_START_TO_BB2]] = cond_br %in0, %mem0_start

  // the branch of the allocation token for the MC
  // CHECK-DAG: %[[MC_MEM0_START_TO_BB1:[^ ]+]], %[[MC_MEM0_START_TO_BB2:[^ ]+]] = cond_br %in0, %mem0_start
  // in bb1 and bb2, the allocation token goes into a lazy fork
  // CHECK-DAG: %[[BB1_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[MC_MEM0_START_TO_BB1]]
  // CHECK-DAG: %[[BB2_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[MC_MEM0_START_TO_BB2]]
  // the bb1 count is 1, for the LSQ store
  // CHECK-DAG: %[[BB1_COUNT:[^ ]+]] = constant %[[BB1_LAZY_FORK]]#0 {{.*}}value = 1 :
  // the bb2 count is 2, for the MC store and the LSQ store
  // CHECK-DAG: %[[BB2_COUNT:[^ ]+]] = constant %[[BB2_LAZY_FORK]]#0 {{.*}}value = 2 :
  // the second outputs of the two lazy forks merge in bb3
  // CHECK-DAG: %[[END_INPUT:[^ ]+]] = merge %[[BB1_LAZY_FORK]]#1, %[[BB2_LAZY_FORK]]#1
  // the MC takes the start of mem0, the bb2 count, the bb1 count,
  // and the output of the merge as its end input
  // CHECK-DAG: mem_controller[%mem0 : memref<4xi32>] start(%mem0_start) groups(bb2 %[[BB2_COUNT]]: [{{.*}}], bb1 %[[BB1_COUNT]]: []) {{.*}}end(%[[END_INPUT]])
  return
}

// -----

// An MC load before a loop nest, and the only LSQ group in the inner loop.
// The non-take-end LSQ (an LSQ connected to an MC) does not need the exit,
// but the exit cannot be pruned: the end of the outer loop decides
// between going back around and leaving, and that decision must stay
// CHECK-LABEL: handshake.func @lsq_in_loop_nest
func.func @lsq_in_loop_nest(%arg0: memref<4xi32> {handshake.arg_name = "mem0"}) -> i32 {
  %c0 = arith.constant 0 : index
  %v0 = memref.load %arg0[%c0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb1(%c0 : index)
^bb1(%i: index):
  %z = arith.constant 0 : index
  cf.br ^bb2(%z : index)
^bb2(%j: index):
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %v = memref.load %arg0[%j] {handshake.name = "load1", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<4xi32>
  memref.store %v, %arg0[%j] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<4xi32>
  %jn = arith.addi %j, %one : index
  %jc = arith.cmpi ult, %jn, %four : index
  cf.cond_br %jc, ^bb2(%jn : index), ^bb3
^bb3:
  %one2 = arith.constant 1 : index
  %four2 = arith.constant 4 : index
  %in = arith.addi %i, %one2 : index
  %ic = arith.cmpi ult, %in, %four2 : index
  cf.cond_br %ic, ^bb1(%in : index), ^bb4
^bb4:
  // the LSQ group is allocated by the first output of a lazy fork
  // CHECK-DAG: lsq[MC] groups(%[[LSQ_LAZY_FORK:[0-9]+]]#0: {{.*}}) mc(
  // in bb2, the allocation token of the LSQ is merged
  // from the start of mem0, the inner loop going back around,
  // and the outer loop going back around
  // CHECK-DAG: %[[LSQ_LAZY_FORK]]:2 = lazy_fork [2] %[[LSQ_TOKEN:[^ ]+]] {handshake.bb = 2
  // CHECK-DAG: %[[LSQ_TOKEN]] = merge %mem0_start, %[[INNER_BACK:[^ ]+]], %[[OUTER_BACK:[^ ]+]] {handshake.bb = 2
  // the second output of the lazy fork goes through the inner loop branch
  // CHECK-DAG: %[[INNER_BACK]], %[[LSQ_TO_BB3:[^ ]+]] = cond_br %{{[^,]+}}, %[[LSQ_LAZY_FORK]]#1
  // and then through the outer loop branch in bb3,
  // whose output leaving the loop is not used
  // CHECK-DAG: %[[OUTER_BACK]], %{{[^ ]+}} = cond_br %{{[^,]+}}, %[[LSQ_TO_BB3]] {handshake.bb = 3
  return %v0 : i32
}
