module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    cf.br ^bb1(%arg0, %c0_i32, %true, %0, %true : i32, i32, i1, i32, i1) {handshake.name = "br0"}
  ^bb1(%1: i32, %2: i32, %3: i1, %4: i32, %5: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %6 = arith.cmpi sle, %2, %1 {handshake.name = "cmpi0"} : i32
    %7 = arith.andi %6, %5 {handshake.name = "andi0"} : i1
    cf.cond_br %7, ^bb2(%1, %3, %4, %2 : i32, i1, i32, i32), ^bb6 {handshake.name = "cond_br0"}
  ^bb2(%8: i32, %9: i1, %10: i32, %11: i32):  // pred: ^bb1
    %12 = arith.addi %11, %8 {handshake.name = "addi0"} : i32
    %13 = arith.shrsi %12, %c1_i32 {handshake.name = "shrsi0"} : i32
    %14 = arith.muli %13, %13 {handshake.name = "muli0"} : i32
    %15 = arith.cmpi eq, %14, %arg0 {handshake.name = "cmpi1"} : i32
    %16 = arith.cmpi ne, %14, %arg0 {handshake.name = "cmpi2"} : i32
    %17 = arith.andi %16, %9 {handshake.name = "andi1"} : i1
    %18 = arith.select %15, %13, %10 {handshake.name = "select0"} : i32
    %19 = arith.cmpi slt, %14, %arg0 {handshake.name = "cmpi3"} : i32
    cf.cond_br %16, ^bb3, ^bb1(%8, %11, %17, %18, %16 : i32, i32, i1, i32, i1)
  ^bb3:  // pred: ^bb2
    cf.cond_br %19, ^bb4, ^bb5 {handshake.name = "cond_br2"}
  ^bb4:  // pred: ^bb3
    %20 = arith.addi %13, %c1_i32 {handshake.name = "addi1"} : i32
    cf.br ^bb1(%8, %20, %17, %18, %16 : i32, i32, i1, i32, i1)
  ^bb5:  // pred: ^bb3
    %21 = arith.addi %13, %c-1_i32 {handshake.name = "addi2"} : i32
    cf.br ^bb1(%21, %11, %17, %18, %16 : i32, i32, i1, i32, i1) {handshake.name = "br7"}
  ^bb6:  // pred: ^bb1
    %22 = arith.select %3, %1, %4 {handshake.name = "select1"} : i32
    return {handshake.name = "return0"} %22 : i32
  }
}

