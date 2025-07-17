module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    %true = arith.constant {handshake.name = "constant0"} true
    %c1_i32 = arith.constant {handshake.name = "constant1"} 1 : i32
    %c-1_i32 = arith.constant {handshake.name = "constant2"} -1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant3"} 0 : i32
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    %1:4 = scf.while (%arg1 = %arg0, %arg2 = %c0_i32, %arg3 = %true, %arg4 = %0, %arg5 = %true) : (i32, i32, i1, i32, i1) -> (i32, i1, i32, i32) {
      %3 = arith.cmpi sle, %arg2, %arg1 {handshake.name = "cmpi0"} : i32
      %4 = arith.andi %3, %arg5 {handshake.name = "andi0"} : i1
      %5 = arith.addi %arg2, %arg1 {handshake.name = "addi0"} : i32
      %6 = arith.shrsi %5, %c1_i32 {handshake.name = "shrsi0"} : i32
      %7 = arith.muli %6, %6 {handshake.name = "muli0"} : i32
      %8 = arith.cmpi ne, %7, %arg0 {handshake.name = "cmpi1"} : i32
      %9 = arith.andi %8, %arg3 {handshake.name = "andi1"} : i1
      %10 = arith.xori %arg5, %true {handshake.name = "xori0"} : i1
      %11 = arith.andi %arg5, %9 {handshake.name = "andi2"} : i1
      %12 = arith.andi %10, %arg3 {handshake.name = "andi3"} : i1
      %13 = arith.ori %11, %12 {handshake.name = "ori0"} : i1
      %14 = arith.xori %4, %true {handshake.name = "xori1"} : i1
      %15 = arith.andi %4, %13 {handshake.name = "andi4"} : i1
      %16 = arith.andi %14, %arg3 {handshake.name = "andi5"} : i1
      %17 = arith.ori %15, %16 {handshake.name = "ori1"} : i1
      scf.condition(%4) {handshake.name = "condition0"} %arg1, %17, %arg4, %arg2 : i32, i1, i32, i32
    } do {
    ^bb0(%arg1: i32, %arg2: i1, %arg3: i32, %arg4: i32):
      %3 = arith.addi %arg4, %arg1 {handshake.name = "addi1"} : i32
      %4 = arith.shrsi %3, %c1_i32 {handshake.name = "shrsi1"} : i32
      %5 = arith.muli %4, %4 {handshake.name = "muli1"} : i32
      %6 = arith.cmpi ne, %5, %arg0 {handshake.name = "cmpi2"} : i32
      %7 = arith.cmpi sle, %arg4, %arg1 {handshake.name = "cmpi3"} : i32
      %8 = arith.cmpi sgt, %arg4, %arg1 {handshake.name = "cmpi4"} : i32
      %9 = arith.cmpi eq, %5, %arg0 {handshake.name = "cmpi5"} : i32
      %10 = arith.andi %7, %6 {handshake.name = "andi6"} : i1
      %11 = arith.select %9, %4, %arg3 {handshake.name = "select0"} : i32
      %12 = arith.ori %10, %8 {handshake.name = "ori2"} : i1
      %13:2 = scf.if %9 -> (i32, i32) {
        scf.yield {handshake.name = "yield0"} %arg1, %arg4 : i32, i32
      } else {
        %14 = arith.cmpi slt, %5, %arg0 {handshake.name = "cmpi6"} : i32
        %15:2 = scf.if %14 -> (i32, i32) {
          %16 = arith.addi %4, %c1_i32 {handshake.name = "addi2"} : i32
          scf.yield {handshake.name = "yield1"} %arg1, %16 : i32, i32
        } else {
          %16 = arith.addi %4, %c-1_i32 {handshake.name = "addi3"} : i32
          scf.yield {handshake.name = "yield2"} %16, %arg4 : i32, i32
        } {handshake.name = "if0"}
        scf.yield {handshake.name = "yield3"} %15#0, %15#1 : i32, i32
      } {handshake.name = "if1"}
      scf.yield {handshake.name = "yield4"} %13#0, %13#1, %arg2, %11, %12 : i32, i32, i1, i32, i1
    } attributes {handshake.name = "while0"}
    %2 = arith.select %1#1, %1#0, %1#2 {handshake.name = "select1"} : i32
    return {handshake.name = "return0"} %2 : i32
  }
}

