module {
  func.func @multiple_exit(%arg0: memref<10xi32> {handshake.arg_name = "arr"}) -> i32 {
    %true = arith.constant {handshake.name = "constant0"} true
    %c2_i32 = arith.constant {handshake.name = "constant1"} 2 : i32
    %c1_i32 = arith.constant {handshake.name = "constant2"} 1 : i32
    %false = arith.constant {handshake.name = "constant3"} false
    %c-1_i32 = arith.constant {handshake.name = "constant4"} -1 : i32
    %c10_i32 = arith.constant {handshake.name = "constant5"} 10 : i32
    %c0_i32 = arith.constant {handshake.name = "constant6"} 0 : i32
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    %1:3 = scf.while (%arg1 = %c0_i32, %arg2 = %true, %arg3 = %0, %arg4 = %true) : (i32, i1, i32, i1) -> (i1, i32, i32) {
      %3 = arith.cmpi slt, %arg1, %c10_i32 {handshake.name = "cmpi0"} : i32
      %4 = arith.andi %3, %arg4 {handshake.name = "andi0"} : i1
      scf.condition(%4) {handshake.name = "condition0"} %arg2, %arg3, %arg1 : i1, i32, i32
    } do {
    ^bb0(%arg1: i1, %arg2: i32, %arg3: i32):
      %3 = arith.index_cast %arg3 {handshake.name = "index_cast0"} : i32 to index
      %4 = memref.load %arg0[%3] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load0"} : memref<10xi32>
      %5 = arith.cmpi ne, %4, %c-1_i32 {handshake.name = "cmpi1"} : i32
      %6:4 = scf.if %5 -> (i1, i32, i1, i32) {
        %7 = memref.load %arg0[%3] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load1"} : memref<10xi32>
        %8 = arith.cmpi eq, %7, %c0_i32 {handshake.name = "cmpi2"} : i32
        %9 = arith.cmpi ne, %7, %c0_i32 {handshake.name = "cmpi3"} : i32
        %10 = arith.andi %9, %arg1 {handshake.name = "andi1"} : i1
        %11 = arith.select %8, %c1_i32, %arg2 {handshake.name = "select0"} : i32
        %12 = scf.if %9 -> (i32) {
          %13 = memref.load %arg0[%3] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load2"} : memref<10xi32>
          %14 = arith.addi %13, %c1_i32 {handshake.name = "addi0"} : i32
          memref.store %14, %arg0[%3] {handshake.deps = #handshake<deps[<"load0" (0)>, <"load1" (0)>, <"load2" (0)>, <"store0" (0)>]>, handshake.name = "store0"} : memref<10xi32>
          %15 = arith.addi %arg3, %c1_i32 {handshake.name = "addi1"} : i32
          scf.yield {handshake.name = "yield0"} %15 : i32
        } else {
          scf.yield {handshake.name = "yield1"} %arg3 : i32
        } {handshake.name = "if0"}
        scf.yield {handshake.name = "yield2"} %10, %11, %9, %12 : i1, i32, i1, i32
      } else {
        scf.yield {handshake.name = "yield3"} %arg1, %arg2, %false, %arg3 : i1, i32, i1, i32
      } {handshake.name = "if1"}
      scf.yield {handshake.name = "yield4"} %6#3, %6#0, %6#1, %6#2 : i32, i1, i32, i1
    } attributes {handshake.name = "while0"}
    %2 = arith.select %1#0, %c2_i32, %1#1 {handshake.name = "select1"} : i32
    return {handshake.name = "return0"} %2 : i32
  }
}

