module {
  func.func @if_convert(%arg0: memref<200xi32> {handshake.arg_name = "a"}, %arg1: memref<200xi32> {handshake.arg_name = "b"}) {
    %c1_i32 = arith.constant {handshake.name = "constant3"} 1 : i32
    cf.br ^bb1(%c1_i32, %c1_i32 : i32, i32) {handshake.name = "br0"}
  ^bb1(%16: i32, %17: i32): // 2 preds: ^bb0, ^bb4
    cf.br ^bb2(%16, %17 : i32, i32) {handshake.name = "br3"}
  ^bb2(%0: i32, %1: i32):  // 2 preds: ^bb1, ^bb3
    %c199_i32 = arith.constant {handshake.name = "constant4"} 199 : i32
    %2 = arith.cmpi slt, %1, %c199_i32 {handshake.name = "cmpi0"} : i32
    %3 = arith.cmpi eq, %1, %0 {handshake.name = "cmpi1"} : i32
    %4 = arith.andi %2, %3 {handshake.name = "andi0"} : i1
    cf.cond_br %4, ^bb3, ^bb4(%0 : i32) {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    %c1_i32_0 = arith.constant {handshake.name = "constant5"} 1 : i32
    %c10000_i32 = arith.constant {handshake.name = "constant6"} 10000 : i32
    %c2_i32 = arith.constant {handshake.name = "constant7"} 2 : i32
    %5 = arith.index_cast %0 {handshake.name = "index_cast0"} : i32 to index
    %6 = memref.load %arg0[%5] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : memref<200xi32>
    %7 = arith.muli %0, %6 {handshake.name = "muli0"} : i32
    %8 = arith.cmpi slt, %7, %c10000_i32 {handshake.name = "cmpi2"} : i32
    %9 = arith.addi %1, %c2_i32 {handshake.name = "addi0"} : i32
    %10 = arith.addi %1, %c1_i32_0 {handshake.name = "addi1"} : i32
    %11 = arith.select %8, %9, %10 {handshake.name = "select1"} : i32
    %12 = arith.index_cast %11 {handshake.name = "index_cast1"} : i32 to index
    memref.store %c1_i32_0, %arg1[%12] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : memref<200xi32>
    %13 = arith.addi %1, %c1_i32_0 {handshake.name = "addi2"} : i32
    cf.br ^bb2(%11, %13 : i32, i32) {handshake.name = "br2"}
  ^bb4(%14: i32):  // pred: ^bb2
    %c199_i32_1 = arith.constant {handshake.name = "constant8"} 199 : i32
    %15 = arith.cmpi slt, %14, %c199_i32_1 {handshake.name = "cmpi3"} : i32
    cf.cond_br %15, ^bb1(%14, %14 : i32, i32), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb3
    return {handshake.name = "return0"}
  }
}

