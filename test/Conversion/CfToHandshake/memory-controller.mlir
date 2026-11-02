// RUN: dynamatic-opt %s --lower-cf-to-handshake --remove-operation-names --split-input-file | FileCheck %s

// An if-else with one store in the true branch
// and two stores in the false branch
// CHECK-LABEL: handshake.func @if_else_stores
func.func @if_else_stores(%arg0: i1 {handshake.arg_name = "in0"},
                          %arg1: memref<4xi32> {handshake.arg_name = "mem0"}) {
  // the start of mem0 produces the allocation token,
  // which goes through a branch on in0 to bb1 or bb2
  // CHECK-DAG: %[[MEM0_START_TO_BB1:[^ ]+]], %[[MEM0_START_TO_BB2:[^ ]+]] = cond_br %in0, %mem0_start
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // in bb1, the allocation token goes into a lazy fork
  // CHECK-DAG: %[[BB1_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[MEM0_START_TO_BB1]]
  // the first output of the bb1 lazy fork triggers the bb1 count, which is 1
  // CHECK-DAG: %[[BB1_COUNT:[^ ]+]] = constant %[[BB1_LAZY_FORK]]#0 {{.*}}value = 1 :
  %i0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  memref.store %c42, %arg1[%i0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb3
^bb2:
  // in bb2, the allocation token goes into a lazy fork
  // CHECK-DAG: %[[BB2_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[MEM0_START_TO_BB2]]
  // the first output of the bb2 lazy fork triggers the bb2 count, which is 2
  // CHECK-DAG: %[[BB2_COUNT:[^ ]+]] = constant %[[BB2_LAZY_FORK]]#0 {{.*}}value = 2 :
  %i1 = arith.constant 1 : index
  %i3 = arith.constant 3 : index
  %c7 = arith.constant 7 : i32
  %c9 = arith.constant 9 : i32
  memref.store %c7, %arg1[%i1] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  memref.store %c9, %arg1[%i3] {handshake.name = "store2", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb3
^bb3:
  // the second outputs of the two lazy forks merge in bb3
  // CHECK-DAG: %[[END_INPUT:[^ ]+]] = merge %[[BB1_LAZY_FORK]]#1, %[[BB2_LAZY_FORK]]#1
  // the MC takes the start of mem0, the bb1 count, the bb2 count,
  // and the output of the merge as its end input
  // CHECK-DAG: mem_controller[%mem0 : memref<4xi32>] start(%mem0_start) groups(bb1 %[[BB1_COUNT]]: [{{.*}}], bb2 %[[BB2_COUNT]]: [{{.*}}]) end(%[[END_INPUT]])
  return
}

// -----

// A loop with one store
// CHECK-LABEL: handshake.func @loop_store
func.func @loop_store(%arg0: memref<4xi32> {handshake.arg_name = "mem0"}) {
  %c0 = arith.constant 0 : index
  cf.br ^bb1(%c0 : index)
^bb1(%i: index):
  // the start of mem0 produces the allocation token,
  // which is merged in bb1 with the allocation token coming back around
  // CHECK-DAG: %[[TOKEN:[^ ]+]] = merge %mem0_start, %[[BB1_LAZY_FORK_TO_BB1:[^ ]+]]
  // in bb1, the allocation token goes into a lazy fork
  // CHECK-DAG: %[[BB1_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[TOKEN]]
  // the first output of the bb1 lazy fork triggers the bb1 count, which is 1
  // CHECK-DAG: %[[BB1_COUNT:[^ ]+]] = constant %[[BB1_LAZY_FORK]]#0 {{.*}}value = 1 :
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c42 = arith.constant 42 : i32
  memref.store %c42, %arg0[%i] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  %next = arith.addi %i, %c1 : index
  // the second output of the bb1 lazy fork is branched on the loop condition:
  // the output to bb1 goes back around to the merge,
  // and the output to bb2 leaves the loop
  // CHECK-DAG: %[[BB1_LAZY_FORK_TO_BB1]], %[[BB1_LAZY_FORK_TO_BB2:[^ ]+]] = cond_br %[[COND:[^,]+]], %[[BB1_LAZY_FORK]]#1
  // CHECK-DAG: %[[COND]] = cmpi ult
  %cond = arith.cmpi ult, %next, %c4 : index
  cf.cond_br %cond, ^bb1(%next : index), ^bb2
^bb2:
  // the MC takes the start of mem0, the bb1 count,
  // and the token leaving the loop as its end input
  // CHECK-DAG: mem_controller[%mem0 : memref<4xi32>] start(%mem0_start) groups(bb1 %[[BB1_COUNT]]: [{{.*}}]) end(%[[BB1_LAZY_FORK_TO_BB2]])
  return
}

// -----

// An if-else with one load in each branch, and no stores
// CHECK-LABEL: handshake.func @if_else_loads
func.func @if_else_loads(%arg0: i1 {handshake.arg_name = "in0"},
                         %arg1: memref<4xi32> {handshake.arg_name = "mem0"}) -> i32 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %i0 = arith.constant 0 : index
  %a = memref.load %arg1[%i0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb3(%a : i32)
^bb2:
  %i1 = arith.constant 1 : index
  %b = memref.load %arg1[%i1] {handshake.name = "load1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb3(%b : i32)
^bb3(%x: i32):
  // with no stores, the MC has no counts,
  // and the start of mem0 goes directly to its end input
  // CHECK-DAG: mem_controller[%mem0 : memref<4xi32>] start(%mem0_start) groups(bb1: [load({{.*}})], bb2: [load({{.*}})]) end(%mem0_start)
  return %x : i32
}

// -----

// Two memories: a store to the first in the true branch of an if-else,
// and a store to the second after the if-else.
// Each memory is routed only through the part of the cfg it needs
// CHECK-LABEL: handshake.func @two_memories
func.func @two_memories(%arg0: i1 {handshake.arg_name = "in0"},
                        %arg1: memref<4xi32> {handshake.arg_name = "mem0"},
                        %arg2: memref<4xi32> {handshake.arg_name = "mem1"}) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %i0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  memref.store %c42, %arg1[%i0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  // the mem0 allocation token is branched on in0
  // CHECK-DAG: %[[MEM0_START_TO_BB1:[^ ]+]], %[[MEM0_START_TO_BB2:[^ ]+]] = cond_br %in0, %mem0_start
  // in bb1, the token goes into a lazy fork
  // CHECK-DAG: %[[MEM0_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %[[MEM0_START_TO_BB1]]
  // the first output of the lazy fork triggers the bb1 count, which is 1
  // CHECK-DAG: %[[MEM0_COUNT:[^ ]+]] = constant %[[MEM0_LAZY_FORK]]#0 {{.*}}value = 1 :
  // the token from bb2, and the second output of the lazy fork, merge in bb3
  // CHECK-DAG: %[[MEM0_END_INPUT:[^ ]+]] = merge %[[MEM0_START_TO_BB2]], %[[MEM0_LAZY_FORK]]#1
  // the mem0 MC takes the bb1 count, and the merge as its end input
  // CHECK-DAG: %[[MEM0_END:[^ ]+]] = mem_controller[%mem0 : memref<4xi32>] start(%mem0_start) groups(bb1 %[[MEM0_COUNT]]: [{{.*}}]) end(%[[MEM0_END_INPUT]])
  %i1 = arith.constant 1 : index
  %c7 = arith.constant 7 : i32
  memref.store %c7, %arg2[%i1] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  // the mem1 allocation token is not branched
  // in bb3, the token goes into a lazy fork
  // CHECK-DAG: %[[MEM1_LAZY_FORK:[0-9]+]]:2 = lazy_fork [2] %mem1_start
  // the first output of the lazy fork triggers the bb3 count, which is 1
  // CHECK-DAG: %[[MEM1_COUNT:[^ ]+]] = constant %[[MEM1_LAZY_FORK]]#0 {{.*}}value = 1 :
  // the mem1 MC takes the bb3 count,
  // and the second output of the lazy fork as its end input
  // CHECK-DAG: %[[MEM1_END:[^ ]+]] = mem_controller[%mem1 : memref<4xi32>] start(%mem1_start) groups(bb3 %[[MEM1_COUNT]]: [{{.*}}]) end(%[[MEM1_LAZY_FORK]]#1)

  // the end takes the end of the mem0 MC, then of the mem1 MC, then the start
  // CHECK-DAG: end {{.*}} %[[MEM0_END]], %[[MEM1_END]], %start
  return
}
