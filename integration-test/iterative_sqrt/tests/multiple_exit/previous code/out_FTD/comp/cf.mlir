module {
  func.func @multiple_exit(%arg0: memref<10xi32> {handshake.arg_name = "arr"}, %arg1: i32 {handshake.arg_name = "size"}) {
    %true = arith.constant {handshake.name = "constant0"} true
    %c1_i32 = arith.constant {handshake.name = "constant1"} 1 : i32
    %false = arith.constant {handshake.name = "constant2"} false
    %c-1_i32 = arith.constant {handshake.name = "constant3"} -1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant4"} 0 : i32
    cf.br ^bb1(%c0_i32, %true : i32, i1) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: i1):  // 2 preds: ^bb0, ^bb10
    %2 = arith.cmpi slt, %0, %arg1 {handshake.name = "cmpi0"} : i32
    %3 = arith.andi %2, %1 {handshake.name = "andi0"} : i1
    cf.cond_br %3, ^bb2(%0 : i32), ^bb11 {handshake.name = "cond_br0"}
  ^bb2(%4: i32):  // pred: ^bb1
    %5 = arith.index_cast %4 {handshake.name = "index_cast0"} : i32 to index
    %6 = memref.load %arg0[%5] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load0"} : memref<10xi32>
    %7 = arith.cmpi ne, %6, %c-1_i32 {handshake.name = "cmpi1"} : i32
    cf.cond_br %7, ^bb3, ^bb8 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %8 = memref.load %arg0[%5] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load1"} : memref<10xi32>
    %9 = arith.cmpi ne, %8, %c0_i32 {handshake.name = "cmpi2"} : i32
    cf.cond_br %9, ^bb4, ^bb5 {handshake.name = "cond_br2"}
  ^bb4:  // pred: ^bb3
    %10 = memref.load %arg0[%5] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load2"} : memref<10xi32>
    %11 = arith.addi %10, %c1_i32 {handshake.name = "addi0"} : i32
    memref.store %11, %arg0[%5] {handshake.deps = #handshake<deps[<"load0" (0)>, <"load1" (0)>, <"load2" (0)>, <"store0" (0)>]>, handshake.name = "store0"} : memref<10xi32>
    %12 = arith.addi %4, %c1_i32 {handshake.name = "addi1"} : i32
    cf.br ^bb6(%12 : i32) {handshake.name = "br1"}
  ^bb5:  // pred: ^bb3
    cf.br ^bb6(%4 : i32) {handshake.name = "br2"}
  ^bb6(%13: i32):  // 2 preds: ^bb4, ^bb5
    cf.br ^bb7 {handshake.name = "br3"}
  ^bb7:  // pred: ^bb6
    cf.br ^bb9(%9, %13 : i1, i32) {handshake.name = "br4"}
  ^bb8:  // pred: ^bb2
    cf.br ^bb9(%false, %4 : i1, i32) {handshake.name = "br5"}
  ^bb9(%14: i1, %15: i32):  // 2 preds: ^bb7, ^bb8
    cf.br ^bb10 {handshake.name = "br6"}
  ^bb10:  // pred: ^bb9
    cf.br ^bb1(%15, %14 : i32, i1) {handshake.name = "br7"}
  ^bb11:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

