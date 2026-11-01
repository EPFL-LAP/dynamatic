// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names | FileCheck %s

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
  // the start of mem0 (%arg2) produces an allocation token for the LSQ
  // and one for the MC, each branched on %arg0 to bb1 or bb2.
  // the two branches look the same, so the LSQ is checked first,
  // and the MC then gets the other branch

  // the LSQ is a non-take-end LSQ (an LSQ connected to an MC),
  // so its groups are allocated directly by the outputs of its branch
  // to bb1 and to bb2, with no lazy forks, and it takes no end input
  // CHECK-DAG: lsq[MC] groups(%[[LSQ_MEM0_START_TO_BB1:[^:]+]]: {{.*}}, %[[LSQ_MEM0_START_TO_BB2:[^:]+]]: {{.*}}) mc(
  // CHECK-DAG: %[[LSQ_MEM0_START_TO_BB1]], %[[LSQ_MEM0_START_TO_BB2]] = cond_br %arg0, %arg2

  // the branch of the allocation token for the MC
  // CHECK-DAG: %[[MC_MEM0_START_TO_BB1:[^ ]+]], %[[MC_MEM0_START_TO_BB2:[^ ]+]] = cond_br %arg0, %arg2
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
  // CHECK-DAG: mem_controller[%arg1 : memref<4xi32>] start(%arg2) groups(bb2 %[[BB2_COUNT]]: [{{.*}}], bb1 %[[BB1_COUNT]]: []) {{.*}}end(%[[END_INPUT]])
  return
}
