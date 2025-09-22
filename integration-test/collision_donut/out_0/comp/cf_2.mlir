module {
  func.func @collision_donut(%arg0: memref<1000xi32> {handshake.arg_name = "x"}, %arg1: memref<1000xi32> {handshake.arg_name = "y"}) -> i32 {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %c4_i32 = arith.constant {handshake.name = "constant1"} 4 : i32
    %c-1_i32 = arith.constant {handshake.name = "constant2"} -1 : i32
    %c19000_i32 = arith.constant {handshake.name = "constant3"} 19000 : i32
    %c-2_i32 = arith.constant {handshake.name = "constant4"} -2 : i32
    %c1_i32 = arith.constant {handshake.name = "constant5"} 1 : i32
    %c1000_i32 = arith.constant {handshake.name = "constant6"} 1000 : i32
    cf.br ^bb1(%c0_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb3
    %1 = arith.extui %0 {handshake.name = "extui0"} : i32 to i64
    %2 = arith.index_cast %1 {handshake.name = "index_cast0"} : i64 to index
    %3 = memref.load %arg0[%2] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : memref<1000xi32>
    %4 = arith.extui %0 {handshake.name = "extui1"} : i32 to i64
    %5 = arith.index_cast %4 {handshake.name = "index_cast1"} : i64 to index
    %6 = memref.load %arg1[%5] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : memref<1000xi32>
    %7 = arith.muli %3, %3 {handshake.name = "muli0"} : i32
    %8 = arith.muli %6, %6 {handshake.name = "muli1"} : i32
    %9 = arith.addi %7, %8 {handshake.name = "addi0"} : i32
    %10 = arith.cmpi ult, %9, %c4_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %10, ^bb4(%0, %c-1_i32 : i32, i32), ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %11 = arith.cmpi ugt, %9, %c19000_i32 {handshake.name = "cmpi1"} : i32
    cf.cond_br %11, ^bb4(%0, %c-2_i32 : i32, i32), ^bb3 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %12 = arith.addi %0, %c1_i32 {handshake.name = "addi1"} : i32
    %13 = arith.cmpi ult, %12, %c1000_i32 {handshake.name = "cmpi2"} : i32
    cf.cond_br %13, ^bb1(%12 : i32), ^bb4(%12, %c0_i32 : i32, i32) {handshake.name = "cond_br2"}
  ^bb4(%14: i32, %15: i32):  // 3 preds: ^bb1, ^bb2, ^bb3
    %16 = arith.shli %14, %c1_i32 {handshake.name = "shli0"} : i32
    %17 = arith.andi %16, %15 {handshake.name = "andi0"} : i32
    return {handshake.name = "return0"} %17 : i32
  }
}

