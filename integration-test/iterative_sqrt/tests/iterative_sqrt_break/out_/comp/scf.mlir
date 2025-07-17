module {
  func.func @iterative_sqrt(%arg0: i32 {handshake.arg_name = "n"}) -> i32 {
    %true = arith.constant {handshake.name = "constant0"} true
    %c1_i32 = arith.constant {handshake.name = "constant1"} 1 : i32
    %c-1_i32 = arith.constant {handshake.name = "constant2"} -1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant3"} 0 : i32
    %0:2 = scf.while (%arg1 = %arg0, %arg2 = %c0_i32, %arg3 = %true) : (i32, i32, i1) -> (i32, i32) {
      %1 = arith.cmpi sle, %arg2, %arg1 {handshake.name = "cmpi0"} : i32
      %2 = arith.andi %1, %arg3 {handshake.name = "andi0"} : i1
      scf.condition(%2) {handshake.name = "condition0"} %arg1, %arg2 : i32, i32
    } do {
    ^bb0(%arg1: i32, %arg2: i32):
      %1 = arith.addi %arg2, %arg1 {handshake.name = "addi0"} : i32
      %2 = arith.shrsi %1, %c1_i32 {handshake.name = "shrsi0"} : i32
      %3 = arith.muli %2, %2 {handshake.name = "muli0"} : i32
      %4 = arith.cmpi ne, %3, %arg0 {handshake.name = "cmpi1"} : i32
      %5 = arith.cmpi sle, %arg2, %arg1 {handshake.name = "cmpi2"} : i32
      %6 = arith.cmpi sgt, %arg2, %arg1 {handshake.name = "cmpi3"} : i32
      %7 = arith.andi %5, %4 {handshake.name = "andi1"} : i1
      %8 = arith.cmpi eq, %3, %arg0 {handshake.name = "cmpi4"} : i32
      %9 = arith.ori %7, %6 {handshake.name = "ori0"} : i1
      %10:2 = scf.if %8 -> (i32, i32) {
        scf.yield {handshake.name = "yield0"} %2, %arg2 : i32, i32
      } else {
        %11 = arith.cmpi slt, %3, %arg0 {handshake.name = "cmpi5"} : i32
        %12:2 = scf.if %11 -> (i32, i32) {
          %13 = arith.addi %2, %c1_i32 {handshake.name = "addi1"} : i32
          scf.yield {handshake.name = "yield1"} %arg1, %13 : i32, i32
        } else {
          %13 = arith.addi %2, %c-1_i32 {handshake.name = "addi2"} : i32
          scf.yield {handshake.name = "yield2"} %13, %arg2 : i32, i32
        } {handshake.name = "if0"}
        scf.yield {handshake.name = "yield3"} %12#0, %12#1 : i32, i32
      } {handshake.name = "if1"}
      scf.yield {handshake.name = "yield4"} %10#0, %10#1, %9 : i32, i32, i1
    } attributes {handshake.name = "while0"}
    return {handshake.name = "return0"} %0#0 : i32
  }
}

