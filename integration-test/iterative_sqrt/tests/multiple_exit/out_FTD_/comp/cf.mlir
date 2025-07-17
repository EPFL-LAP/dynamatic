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
    cf.br ^bb1(%c0_i32, %true, %0, %true : i32, i1, i32, i1) {handshake.name = "br0"}
  ^bb1(%1: i32, %2: i1, %3: i32, %4: i1):  // 2 preds: ^bb0, ^bb10
    %5 = arith.cmpi slt, %1, %c10_i32 {handshake.name = "cmpi0"} : i32
    %6 = arith.andi %5, %4 {handshake.name = "andi0"} : i1
    cf.cond_br %6, ^bb2(%2, %3, %1 : i1, i32, i32), ^bb11 {handshake.name = "cond_br0"}
  ^bb2(%7: i1, %8: i32, %9: i32):  // pred: ^bb1
    %10 = arith.index_cast %9 {handshake.name = "index_cast0"} : i32 to index
    %11 = memref.load %arg0[%10] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load0"} : memref<10xi32>
    %12 = arith.cmpi ne, %11, %c-1_i32 {handshake.name = "cmpi1"} : i32
    cf.cond_br %12, ^bb3, ^bb8 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %13 = memref.load %arg0[%10] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load1"} : memref<10xi32>
    %14 = arith.cmpi eq, %13, %c0_i32 {handshake.name = "cmpi2"} : i32
    %15 = arith.cmpi ne, %13, %c0_i32 {handshake.name = "cmpi3"} : i32
    %16 = arith.andi %15, %7 {handshake.name = "andi1"} : i1
    %17 = arith.select %14, %c1_i32, %8 {handshake.name = "select0"} : i32
    cf.cond_br %15, ^bb4, ^bb5 {handshake.name = "cond_br2"}
  ^bb4:  // pred: ^bb3
    %18 = memref.load %arg0[%10] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load2"} : memref<10xi32>
    %19 = arith.addi %18, %c1_i32 {handshake.name = "addi0"} : i32
    memref.store %19, %arg0[%10] {handshake.deps = #handshake<deps[<"load0" (0)>, <"load1" (0)>, <"load2" (0)>, <"store0" (0)>]>, handshake.name = "store0"} : memref<10xi32>
    %20 = arith.addi %9, %c1_i32 {handshake.name = "addi1"} : i32
    cf.br ^bb6(%20 : i32) {handshake.name = "br1"}
  ^bb5:  // pred: ^bb3
    cf.br ^bb6(%9 : i32) {handshake.name = "br2"}
  ^bb6(%21: i32):  // 2 preds: ^bb4, ^bb5
    cf.br ^bb7 {handshake.name = "br3"}
  ^bb7:  // pred: ^bb6
    cf.br ^bb9(%16, %17, %15, %21 : i1, i32, i1, i32) {handshake.name = "br4"}
  ^bb8:  // pred: ^bb2
    cf.br ^bb9(%7, %8, %false, %9 : i1, i32, i1, i32) {handshake.name = "br5"}
  ^bb9(%22: i1, %23: i32, %24: i1, %25: i32):  // 2 preds: ^bb7, ^bb8
    cf.br ^bb10 {handshake.name = "br6"}
  ^bb10:  // pred: ^bb9
    cf.br ^bb1(%25, %22, %23, %24 : i32, i1, i32, i1) {handshake.name = "br7"}
  ^bb11:  // pred: ^bb1
    %26 = arith.select %2, %c2_i32, %3 {handshake.name = "select1"} : i32
    return {handshake.name = "return0"} %26 : i32
  }
}

