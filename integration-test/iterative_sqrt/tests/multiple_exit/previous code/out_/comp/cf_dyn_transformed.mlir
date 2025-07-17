module {
  func.func @multiple_exit(%arg0: memref<10xi32> {handshake.arg_name = "arr"}, %arg1: i32 {handshake.arg_name = "size"}) {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %true = arith.constant {handshake.name = "constant4"} true
    cf.br ^bb1(%c0_i32, %true : i32, i1) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
    %2 = arith.cmpi slt, %0, %arg1 {handshake.name = "cmpi0"} : i32
    %3 = arith.andi %2, %1 {handshake.name = "andi0"} : i1
    cf.cond_br %3, ^bb2(%0 : i32), ^bb5 {handshake.name = "cond_br0"}
  ^bb2(%4: i32):  // pred: ^bb1
    %false = arith.constant {handshake.name = "constant5"} false
    %c-1_i32 = arith.constant {handshake.name = "constant6"} -1 : i32
    %5 = arith.index_cast %4 {handshake.name = "index_cast0"} : i32 to index
    %6 = memref.load %arg0[%5] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : memref<10xi32>
    %7 = arith.cmpi ne, %6, %c-1_i32 {handshake.name = "cmpi1"} : i32
    cf.cond_br %7, ^bb3, ^bb1(%4, %false : i32, i1) {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %c0_i32_0 = arith.constant {handshake.name = "constant7"} 0 : i32
    %8 = memref.load %arg0[%5] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load1"} : memref<10xi32>
    %9 = arith.cmpi ne, %8, %c0_i32_0 {handshake.name = "cmpi2"} : i32
    cf.cond_br %9, ^bb4, ^bb1(%4, %9 : i32, i1) {handshake.name = "cond_br2"}
  ^bb4:  // pred: ^bb3
    %true_1 = arith.constant {handshake.name = "constant8"} true
    %c1_i32 = arith.constant {handshake.name = "constant9"} 1 : i32
    %10 = memref.load %arg0[%5] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load2"} : memref<10xi32>
    %11 = arith.addi %10, %c1_i32 {handshake.name = "addi0"} : i32
    memref.store %11, %arg0[%5] {handshake.deps = #handshake<deps[<"load0" (0)>, <"load1" (0)>, <"load2" (0)>, <"store0" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : memref<10xi32>
    %12 = arith.addi %4, %c1_i32 {handshake.name = "addi1"} : i32
    cf.br ^bb1(%12, %true_1 : i32, i1) {handshake.name = "br7"}
  ^bb5:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

