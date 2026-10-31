// RUN: dynamatic-opt --func-merge-returns --lower-cf-to-handshake --remove-operation-names %s --split-input-file | FileCheck %s

// one group: the memory start goes through a lazy fork,
// one output allocates the group, the other is the LSQ's end
// CHECK-LABEL: handshake.func @simpleOneGroupLSQ(
// CHECK-SAME:    %[[MEM0:[a-zA-Z0-9_]+]]: memref<64xi32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         lsq{{\[}}%[[MEM0]] : memref<64xi32>] start(%[[MEM0_START]]) groups(%[[FORK:[0-9]+]]#0: [load({{.*}}), store({{.*}}), store({{.*}})]) end(%[[FORK]]#1)
// CHECK:         %[[FORK]]:2 = lazy_fork [2] %[[MEM0_START]] {handshake.bb = 0 : ui32}
func.func @simpleOneGroupLSQ(%mem: memref<64xi32> {handshake.arg_name = "mem0"}) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ldData1 = memref.load %mem[%c0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  memref.store %ldData1, %mem[%c1] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  memref.store %ldData1, %mem[%c2] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  return %ldData1 : i32
}

// -----

// two groups in two blocks: the memory start goes through
// one lazy fork per block, in program order,
// and the last fork's continuing output is the LSQ's end
// CHECK-LABEL: handshake.func @simpleMultiGroupLSQ(
// CHECK-SAME:    %[[MEM0:[a-zA-Z0-9_]+]]: memref<64xi32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         lsq{{\[}}%[[MEM0]] : memref<64xi32>] start(%[[MEM0_START]]) groups(%[[FORK0:[0-9]+]]#0: [load({{.*}}), load({{.*}})], %[[FORK1:[0-9]+]]#0: [store({{.*}}), store({{.*}})]) end(%[[FORK1]]#1)
// CHECK:         %[[FORK0]]:2 = lazy_fork [2] %[[MEM0_START]] {handshake.bb = 0 : ui32}
// CHECK:         %[[FORK1]]:2 = lazy_fork [2] %[[FORK0]]#1 {handshake.bb = 1 : ui32
func.func @simpleMultiGroupLSQ(%mem: memref<64xi32> {handshake.arg_name = "mem0"}) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ldData1 = memref.load %mem[%c0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  %ldData2 = memref.load %mem[%c1] {handshake.name = "load1", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  cf.br ^bb1(%ldData1, %ldData2 : i32, i32)
^bb1(%stData1: i32, %stData2: i32):
  memref.store %stData1, %mem[%c1] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<LSQ: 1>} : memref<64xi32>
  memref.store %stData2, %mem[%c2] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<LSQ: 1>} : memref<64xi32>
  return %stData1 : i32
}

// -----

// an LSQ next to an MC: the MC takes the end,
// so the LSQ's last group needs no lazy fork
// and is allocated directly by the first fork's continuing output
// CHECK-LABEL: handshake.func @mixLSQAndMCLoads(
// CHECK-SAME:    %[[MEM0:[a-zA-Z0-9_]+]]: memref<64xi32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         mem_controller{{\[}}%[[MEM0]] : memref<64xi32>] start(%[[MEM0_START]]) groups(bb0: [load({{.*}})], bb1: [load({{.*}})]) lsq({{.*}}) end(%[[MEM0_START]])
// CHECK:         lsq[MC] groups(%[[FORK:[0-9]+]]#0: [load({{.*}})], %[[FORK]]#1: [load({{.*}})])
// CHECK:         %[[FORK]]:2 = lazy_fork [2] %[[MEM0_START]] {handshake.bb = 0 : ui32}
// CHECK-NOT:     lazy_fork
func.func @mixLSQAndMCLoads(%mem: memref<64xi32> {handshake.arg_name = "mem0"}) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ldData1 = memref.load %mem[%c0] {handshake.name = "load0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  %ldData2 = memref.load %mem[%c1] {handshake.name = "load1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<64xi32>
  cf.br ^bb1
^bb1:
  %ldData3 = memref.load %mem[%c0] {handshake.name = "load2", handshake.mem_interface = #handshake.mem_interface<LSQ: 1>} : memref<64xi32>
  %ldData4 = memref.load %mem[%c2] {handshake.name = "load3", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<64xi32>
  %add = arith.addi %ldData3, %ldData4 : i32
  return %add : i32
}

// -----

// an LSQ next to an MC with stores: the MC counts the stores
// of each block, including the LSQ's, and takes the end
// CHECK-LABEL: handshake.func @mixLSQAndMCStores(
// CHECK-SAME:    %[[MEM0:[a-zA-Z0-9_]+]]: memref<64xi32>, %{{[a-zA-Z0-9_]+}}: !handshake.channel<i32>, %[[MEM0_START:[a-zA-Z0-9_]+]]: !handshake.control<>
// CHECK:         mem_controller{{\[}}%[[MEM0]] : memref<64xi32>] start(%[[MEM0_START]]) groups(bb0 %[[COUNT0:[0-9]+]]: [store({{.*}})], bb1 %[[COUNT1:[0-9]+]]: []) lsq({{.*}}) end(%[[MC_FORK1:[0-9]+]]#1)
// CHECK:         lsq[MC] groups(%[[LSQ_FORK:[0-9]+]]#0: [store({{.*}})], %[[LSQ_FORK]]#1: [store({{.*}})])
// CHECK:         %[[LSQ_FORK]]:2 = lazy_fork [2] %[[MEM0_START]] {handshake.bb = 0 : ui32}
// CHECK:         %[[MC_FORK0:[0-9]+]]:2 = lazy_fork [2] %[[MEM0_START]] {handshake.bb = 0 : ui32}
// CHECK:         %[[COUNT0]] = constant %[[MC_FORK0]]#0 {handshake.bb = 0 : ui32, value = 2 : i32}
// CHECK:         %[[MC_FORK1]]:2 = lazy_fork [2] %[[MC_FORK0]]#1 {handshake.bb = 1 : ui32
// CHECK:         %[[COUNT1]] = constant %[[MC_FORK1]]#0 {handshake.bb = 1 : ui32, value = 1 : i32}
func.func @mixLSQAndMCStores(%mem: memref<64xi32> {handshake.arg_name = "mem0"}, %data : i32 {handshake.arg_name = "in0"}) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  memref.store %data, %mem[%c0] {handshake.name = "store0", handshake.mem_interface = #handshake.mem_interface<LSQ: 0>} : memref<64xi32>
  memref.store %data, %mem[%c1] {handshake.name = "store1", handshake.mem_interface = #handshake.mem_interface<MC>} : memref<64xi32>
  cf.br ^bb1(%data : i32)
^bb1(%stData : i32):
  memref.store %stData, %mem[%c2] {handshake.name = "store2", handshake.mem_interface = #handshake.mem_interface<LSQ: 1>} : memref<64xi32>
  return %stData : i32
}
