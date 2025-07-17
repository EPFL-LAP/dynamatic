module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    %true = arith.constant {handshake.name = "constant0"} true
    %c1_i32 = arith.constant {handshake.name = "constant1"} 1 : i32
    %c-1_i32 = arith.constant {handshake.name = "constant2"} -1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant3"} 0 : i32
    cf.br ^bb1(%arg0, %c0_i32, %true : i32, i32, i1) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: i32, %2: i1):  // 2 preds: ^bb0, ^bb10
    %3 = arith.cmpi sle, %1, %0 {handshake.name = "cmpi0"} : i32
    %4 = arith.andi %3, %2 {handshake.name = "andi0"} : i1
    cf.cond_br %4, ^bb2(%0, %1 : i32, i32), ^bb11 {handshake.name = "cond_br0"}
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
    cf.cond_br %14, ^bb3, ^bb4 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    cf.br ^bb9(%8, %6 : i32, i32) {handshake.name = "br1"}
  ^bb4:  // pred: ^bb2
    %16 = arith.cmpi slt, %9, %arg0 {handshake.name = "cmpi5"} : i32
    cf.cond_br %16, ^bb5, ^bb6 {handshake.name = "cond_br2"}
  ^bb5:  // pred: ^bb4
    %17 = arith.addi %8, %c1_i32 {handshake.name = "addi1"} : i32
    cf.br ^bb7(%5, %17 : i32, i32) {handshake.name = "br2"}
  ^bb6:  // pred: ^bb4
    %18 = arith.addi %8, %c-1_i32 {handshake.name = "addi2"} : i32
    cf.br ^bb7(%18, %6 : i32, i32) {handshake.name = "br3"}
  ^bb7(%19: i32, %20: i32):  // 2 preds: ^bb5, ^bb6
    cf.br ^bb8 {handshake.name = "br4"}
  ^bb8:  // pred: ^bb7
    cf.br ^bb9(%19, %20 : i32, i32) {handshake.name = "br5"}
  ^bb9(%21: i32, %22: i32):  // 2 preds: ^bb3, ^bb8
    cf.br ^bb10 {handshake.name = "br6"}
  ^bb10:  // pred: ^bb9
    cf.br ^bb1(%21, %22, %15 : i32, i32, i1) {handshake.name = "br7"}
  ^bb11:  // pred: ^bb1
    return {handshake.name = "return0"} %0 : i32
  }
}

