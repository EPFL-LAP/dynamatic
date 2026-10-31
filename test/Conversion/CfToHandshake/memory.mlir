// RUN: dynamatic-opt --func-merge-returns --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// one block with a store and a load: the memory start goes through a lazy fork,
// one output triggers the block's store count, the other is the MC's end
// CHECK-LABEL: handshake.func @simpleLoadStore(
// CHECK-SAME:    %[[MEM0:[a-zA-Z0-9_]+]]: memref<4xi32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         mem_controller{{\[}}%[[MEM0]] : memref<4xi32>] start(%[[MEM0_START]]) groups(bb0 %[[COUNT:[0-9]+]]: [store({{.*}}), load({{.*}})]) end(%[[FORK:[0-9]+]]#1)
// CHECK:         %[[FORK]]:2 = lazy_fork [2] %[[MEM0_START]] {handshake.bb = 0 : ui32}
// CHECK:         %[[COUNT]] = constant %[[FORK]]#0 {handshake.bb = 0 : ui32, value = 1 : i32}
func.func @simpleLoadStore(%arg0 : index {handshake.arg_name = "in0"}, %arg1 : memref<4xi32> {handshake.arg_name = "mem0"}) {
  %c1 = arith.constant 1 : i32
  memref.store %c1, %arg1[%arg0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  %0 = memref.load %arg1[%arg0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  return
}

// -----

// stores in both branches: the memory start is branched to each store block,
// and the two continuing outputs merge into the MC's end
// CHECK-LABEL: handshake.func @storeMulBlocks(
// CHECK-SAME:    %[[COND:[a-zA-Z0-9_]+]]: !handshake.channel<i1>, %{{[a-zA-Z0-9_]+}}: !handshake.channel<i32>, %[[MEM0:[a-zA-Z0-9_]+]]: memref<4xi32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         mem_controller{{\[}}%[[MEM0]] : memref<4xi32>] start(%[[MEM0_START]]) groups(bb1 %[[COUNT1:[0-9]+]]: [store({{.*}})], bb2 %[[COUNT2:[0-9]+]]: [store({{.*}})]) end(%[[END:[0-9]+]])
// CHECK:         %[[TRUE:[a-zA-Z0-9_]+]], %[[FALSE:[a-zA-Z0-9_]+]] = cond_br %[[COND]], %[[MEM0_START]] {handshake.bb = 0 : ui32}
// CHECK:         %[[FORK1:[0-9]+]]:2 = lazy_fork [2] %[[TRUE]] {handshake.bb = 1 : ui32}
// CHECK:         %[[COUNT1]] = constant %[[FORK1]]#0 {handshake.bb = 1 : ui32, value = 1 : i32}
// CHECK:         %[[FORK2:[0-9]+]]:2 = lazy_fork [2] %[[FALSE]] {handshake.bb = 2 : ui32}
// CHECK:         %[[COUNT2]] = constant %[[FORK2]]#0 {handshake.bb = 2 : ui32, value = 1 : i32}
// CHECK:         %[[END]] = merge %[[FORK1]]#1, %[[FORK2]]#1 {handshake.bb = 3 : ui32
func.func @storeMulBlocks(%arg0 : i1 {handshake.arg_name = "in0"}, %arg1 : index {handshake.arg_name = "in1"}, %arg2 : memref<4xi32> {handshake.arg_name = "mem0"}) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  memref.store %c1, %arg2[%arg1] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb3
^bb2:
  %c2 = arith.constant 2 : i32
  memref.store %c2, %arg2[%arg1] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.br ^bb3
^bb3:
  return
}

// -----

// an MC with only loads takes the memory start as its end,
// and the loaded value is routed to the block that uses it
// CHECK-LABEL: handshake.func @forwardLoadToBB(
// CHECK-SAME:    %[[COND:[a-zA-Z0-9_]+]]: !handshake.channel<i1>, %{{[a-zA-Z0-9_]+}}: !handshake.channel<i32>, %[[MEM0:[a-zA-Z0-9_]+]]: memref<4xi32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         mem_controller{{\[}}%[[MEM0]] : memref<4xi32>] start(%[[MEM0_START]]) groups(bb0: [load({{.*}})]) end(%[[MEM0_START]])
// CHECK:         %{{[a-zA-Z0-9_]+}}, %[[DATA:[a-zA-Z0-9_]+]] = load{{.*}} {handshake.bb = 0 : ui32
// CHECK:         %[[TRUE:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = cond_br %[[COND]], %[[DATA]] {handshake.bb = 0 : ui32}
// CHECK:         addi %[[TRUE]], {{.*}} {handshake.bb = 1 : ui32
func.func @forwardLoadToBB(%arg0 : i1 {handshake.arg_name = "in0"}, %arg1 : index {handshake.arg_name = "in1"}, %arg2: memref<4xi32> {handshake.arg_name = "mem0"}) {
  %0 = memref.load %arg2[%arg1] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %1 = arith.addi %0, %c1 : i32
  cf.br ^bb2
^bb2:
  return
}

// -----

// two memories each get their own MC, with their own start routed
// to their own store block, and their own end
// CHECK-LABEL: handshake.func @multipleMemories(
// CHECK-SAME:    %[[MEM0:[a-zA-Z0-9_]+]]: memref<4xi32>, %[[MEM1:[a-zA-Z0-9_]+]]: memref<4xi32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>, %[[MEM1_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         %{{[a-zA-Z0-9_]+}}, %[[MEM1_END:[a-zA-Z0-9_]+]] = mem_controller{{\[}}%[[MEM1]] : memref<4xi32>] start(%[[MEM1_START]]) groups(bb1: [load({{.*}})], bb2 %{{[0-9]+}}: [store({{.*}})]) end(%{{[0-9]+}})
// CHECK:         %{{[a-zA-Z0-9_]+}}, %[[MEM0_END:[a-zA-Z0-9_]+]] = mem_controller{{\[}}%[[MEM0]] : memref<4xi32>] start(%[[MEM0_START]]) groups(bb1 %{{[0-9]+}}: [store({{.*}})], bb2: [load({{.*}})]) end(%{{[0-9]+}})
// CHECK:         end {handshake.bb = 3 : ui32} %[[MEM0_END]], %[[MEM1_END]],
func.func @multipleMemories(%arg0 : i1 {handshake.arg_name = "in0"}, %arg1: memref<4xi32> {handshake.arg_name = "mem0"}, %arg2: memref<4xi32> {handshake.arg_name = "mem1"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %1 = memref.load %arg2[%c0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  memref.store %1, %arg1[%c0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  return
^bb2:
  %2 = memref.load %arg1[%c1] {handshake.name = "load1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  memref.store %2, %arg2[%c1] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<4xi32>
  return
}
