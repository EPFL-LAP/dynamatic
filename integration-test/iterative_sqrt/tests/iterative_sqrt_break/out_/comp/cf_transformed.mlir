module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    cf.br ^bb1(%arg0, %c0_i32, %true : i32, i32, i1) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: i32, %2: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %3 = arith.cmpi sle, %1, %0 {handshake.name = "cmpi0"} : i32
    %4 = arith.andi %3, %2 {handshake.name = "andi0"} : i1
    cf.cond_br %4, ^bb2(%0, %1 : i32, i32), ^bb6 {handshake.name = "cond_br0"}
  ^bb2(%5: i32, %6: i32):  // pred: ^bb1
    %7 = arith.addi %6, %5 {handshake.name = "addi0"} : i32
    %8 = arith.shrsi %7, %c1_i32 {handshake.name = "shrsi0"} : i32
    %9 = arith.muli %8, %8 {handshake.name = "muli0"} : i32
    %10 = arith.cmpi ne, %9, %arg0 {handshake.name = "cmpi1"} : i32
    %11 = arith.cmpi sle, %6, %5 {handshake.name = "cmpi2"} : i32
    %12 = arith.cmpi sgt, %6, %5 {handshake.name = "cmpi3"} : i32
    %13 = arith.andi %11, %10 {handshake.name = "andi1"} : i1
    %14 = arith.cmpi eq, %9, %arg0 {handshake.name = "cmpi4"} : i32
    %15 = arith.ori %13, %12 {handshake.name = "ori0"} : i1
    cf.cond_br %14, ^bb1(%8, %6, %15 : i32, i32, i1), ^bb3
  ^bb3:  // pred: ^bb2
    %16 = arith.cmpi slt, %9, %arg0 {handshake.name = "cmpi5"} : i32
    cf.cond_br %16, ^bb4, ^bb5 {handshake.name = "cond_br2"}
  ^bb4:  // pred: ^bb3
    %17 = arith.addi %8, %c1_i32 {handshake.name = "addi1"} : i32
    cf.br ^bb1(%5, %17, %15 : i32, i32, i1)
  ^bb5:  // pred: ^bb3
    %18 = arith.addi %8, %c-1_i32 {handshake.name = "addi2"} : i32
    cf.br ^bb1(%18, %6, %15 : i32, i32, i1) {handshake.name = "br7"}
  ^bb6:  // pred: ^bb1
    return {handshake.name = "return0"} %0 : i32
  }
}

