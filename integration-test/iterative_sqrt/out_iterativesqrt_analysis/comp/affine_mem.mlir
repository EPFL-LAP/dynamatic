module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    %true = arith.constant {handshake.name = "constant0"} true
    %c-1_i32 = arith.constant {handshake.name = "constant1"} -1 : i32
    %c1_i32 = arith.constant {handshake.name = "constant2"} 1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant3"} 0 : i32
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    %1:4 = scf.while (%arg1 = %arg0, %arg2 = %c0_i32, %arg3 = %true, %arg4 = %0, %arg5 = %true) : (i32, i32, i1, i32, i1) -> (i32, i1, i32, i32) {
      %3 = arith.cmpi sle, %arg2, %arg1 {handshake.name = "cmpi0"} : i32
      %4 = arith.andi %3, %arg5 {handshake.name = "andi0"} : i1
      scf.condition(%4) {handshake.name = "condition0"} %arg1, %arg3, %arg4, %arg2 : i32, i1, i32, i32
    } do {
    ^bb0(%arg1: i32, %arg2: i1, %arg3: i32, %arg4: i32):
      %3 = arith.addi %arg4, %arg1 {handshake.name = "addi0"} : i32
      %4 = arith.shrsi %3, %c1_i32 {handshake.name = "shrsi0"} : i32
      %5 = arith.muli %4, %4 {handshake.name = "muli0"} : i32
      %6 = arith.cmpi eq, %5, %arg0 {handshake.name = "cmpi1"} : i32
      %7 = arith.cmpi ne, %5, %arg0 {handshake.name = "cmpi2"} : i32
      %8 = arith.andi %7, %arg2 {handshake.name = "andi1"} : i1
      %9 = arith.select %6, %4, %arg3 {handshake.name = "select0"} : i32
      %10 = arith.cmpi slt, %5, %arg0 {handshake.name = "cmpi3"} : i32
      %11:2 = scf.if %7 -> (i32, i32) {
        %12:2 = scf.if %10 -> (i32, i32) {
          %13 = arith.addi %4, %c1_i32 {handshake.name = "addi1"} : i32
          scf.yield {handshake.name = "yield0"} %arg1, %13 : i32, i32
        } else {
          %13 = arith.addi %4, %c-1_i32 {handshake.name = "addi2"} : i32
          scf.yield {handshake.name = "yield1"} %13, %arg4 : i32, i32
        } {handshake.name = "if0"}
        scf.yield {handshake.name = "yield2"} %12#0, %12#1 : i32, i32
      } else {
        scf.yield {handshake.name = "yield3"} %arg1, %arg4 : i32, i32
      } {handshake.name = "if1"}
      scf.yield {handshake.name = "yield4"} %11#0, %11#1, %8, %9, %7 : i32, i32, i1, i32, i1
    } attributes {handshake.name = "while0"}
    %2 = arith.select %1#1, %1#0, %1#2 {handshake.name = "select1"} : i32
    return {handshake.name = "return0"} %2 : i32
  }
}

