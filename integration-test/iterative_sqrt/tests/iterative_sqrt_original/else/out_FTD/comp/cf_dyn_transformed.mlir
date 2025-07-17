module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %true = arith.constant {handshake.name = "constant3"} true
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    cf.br ^bb1(%arg0, %c0_i32, %true, %0, %true : i32, i32, i1, i32, i1) {handshake.name = "br0"}
  ^bb1(%1: i32, %2: i32, %3: i1, %4: i32, %5: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
    %true_0 = arith.constant {handshake.name = "constant4"} true
    %c1_i32 = arith.constant {handshake.name = "constant5"} 1 : i32
    %6 = arith.cmpi sle, %2, %1 {handshake.name = "cmpi0"} : i32
    %7 = arith.andi %6, %5 {handshake.name = "andi0"} : i1
    %8 = arith.addi %2, %1 {handshake.name = "addi0"} : i32
    %9 = arith.shrsi %8, %c1_i32 {handshake.name = "shrsi0"} : i32
    %10 = arith.muli %9, %9 {handshake.name = "muli0"} : i32
    %11 = arith.cmpi ne, %10, %arg0 {handshake.name = "cmpi1"} : i32
    %12 = arith.andi %11, %3 {handshake.name = "andi1"} : i1
    %13 = arith.xori %5, %true_0 {handshake.name = "xori0"} : i1
    %14 = arith.andi %5, %12 {handshake.name = "andi2"} : i1
    %15 = arith.andi %13, %3 {handshake.name = "andi3"} : i1
    %16 = arith.ori %14, %15 {handshake.name = "ori0"} : i1
    %17 = arith.xori %7, %true_0 {handshake.name = "xori1"} : i1
    %18 = arith.andi %7, %16 {handshake.name = "andi4"} : i1
    %19 = arith.andi %17, %3 {handshake.name = "andi5"} : i1
    %20 = arith.ori %18, %19 {handshake.name = "ori1"} : i1
    cf.cond_br %7, ^bb2(%1, %20, %4, %2 : i32, i1, i32, i32), ^bb6 {handshake.name = "cond_br0"}
  ^bb2(%21: i32, %22: i1, %23: i32, %24: i32):  // pred: ^bb1
    %c1_i32_1 = arith.constant {handshake.name = "constant6"} 1 : i32
    %25 = arith.addi %24, %21 {handshake.name = "addi1"} : i32
    %26 = arith.shrsi %25, %c1_i32_1 {handshake.name = "shrsi1"} : i32
    %27 = arith.muli %26, %26 {handshake.name = "muli1"} : i32
    %28 = arith.cmpi ne, %27, %arg0 {handshake.name = "cmpi2"} : i32
    %29 = arith.cmpi sle, %24, %21 {handshake.name = "cmpi3"} : i32
    %30 = arith.cmpi sgt, %24, %21 {handshake.name = "cmpi4"} : i32
    %31 = arith.cmpi eq, %27, %arg0 {handshake.name = "cmpi5"} : i32
    %32 = arith.andi %29, %28 {handshake.name = "andi6"} : i1
    %33 = arith.select %31, %26, %23 {handshake.name = "select0"} : i32
    %34 = arith.ori %32, %30 {handshake.name = "ori2"} : i1
    cf.cond_br %31, ^bb1(%21, %24, %22, %33, %34 : i32, i32, i1, i32, i1), ^bb3 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %35 = arith.cmpi slt, %27, %arg0 {handshake.name = "cmpi6"} : i32
    cf.cond_br %35, ^bb4, ^bb5 {handshake.name = "cond_br2"}
  ^bb4:  // pred: ^bb3
    %c1_i32_2 = arith.constant {handshake.name = "constant7"} 1 : i32
    %36 = arith.addi %26, %c1_i32_2 {handshake.name = "addi2"} : i32
    cf.br ^bb1(%21, %36, %22, %33, %34 : i32, i32, i1, i32, i1) {handshake.name = "br1"}
  ^bb5:  // pred: ^bb3
    %c-1_i32 = arith.constant {handshake.name = "constant8"} -1 : i32
    %37 = arith.addi %26, %c-1_i32 {handshake.name = "addi3"} : i32
    cf.br ^bb1(%37, %24, %22, %33, %34 : i32, i32, i1, i32, i1) {handshake.name = "br7"}
  ^bb6:  // pred: ^bb1
    %38 = arith.select %20, %1, %4 {handshake.name = "select1"} : i32
    return {handshake.name = "return0"} %38 : i32
  }
}

