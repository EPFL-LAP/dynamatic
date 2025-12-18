module {
  func.func @jacobi_1d_imper(%arg0: memref<100xi32> {handshake.arg_name = "A"}, %arg1: memref<100xi32> {handshake.arg_name = "B"}) {
    %c-1 = arith.constant {handshake.name = "constant7"} -1 : index
    %c99 = arith.constant {handshake.name = "constant5"} 99 : index
    %c3_i32 = arith.constant {handshake.name = "constant0"} 3 : i32
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %c3 = arith.constant {handshake.name = "constant2"} 3 : index
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    cf.br ^bb2(%c1 : index) {handshake.name = "br1"}
  ^bb2(%1: index):  // 2 preds: ^bb1, ^bb2
    %2 = arith.addi %1, %c-1 {handshake.name = "addi2"} : index
    %3 = memref.load %arg0[%2] {handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.name = "load0"} : memref<100xi32>
    %4 = memref.load %arg0[%1] {handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.name = "load1"} : memref<100xi32>
    %5 = arith.addi %3, %4 {handshake.name = "addi0"} : i32
    %6 = arith.addi %1, %c1 {handshake.name = "addi3"} : index
    %7 = memref.load %arg0[%6] {handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.name = "load2"} : memref<100xi32>
    %8 = arith.addi %5, %7 {handshake.name = "addi1"} : i32
    %9 = arith.muli %8, %c3_i32 {handshake.name = "muli0"} : i32
    memref.store %9, %arg1[%1] {handshake.deps = #handshake<deps[["store0", 1], ["load3", 1], ["load3", 2]]>, handshake.name = "store0"} : memref<100xi32>
    %10 = arith.addi %1, %c1 {handshake.name = "addi4"} : index
    %11 = arith.cmpi ult, %10, %c99 {handshake.name = "cmpi0"} : index
    cf.cond_br %11, ^bb2(%10 : index), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    cf.br ^bb4(%c1 : index) {handshake.name = "br2"}
  ^bb4(%12: index):  // 2 preds: ^bb3, ^bb4
    %13 = memref.load %arg1[%12] {handshake.deps = #handshake<deps[["store0", 1]]>, handshake.name = "load3"} : memref<100xi32>
    memref.store %13, %arg0[%12] {handshake.deps = #handshake<deps[["load0", 1], ["load1", 1], ["load2", 1], ["store1", 1]]>, handshake.name = "store1"} : memref<100xi32>
    %14 = arith.addi %12, %c1 {handshake.name = "addi5"} : index
    %15 = arith.cmpi ult, %14, %c99 {handshake.name = "cmpi1"} : index
    cf.cond_br %15, ^bb4(%14 : index), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb4
    %16 = arith.addi %0, %c1 {handshake.name = "addi6"} : index
    %17 = arith.cmpi ult, %16, %c3 {handshake.name = "cmpi2"} : index
    cf.cond_br %17, ^bb1(%16 : index), ^bb6 {handshake.name = "cond_br2"}
  ^bb6:  // pred: ^bb5
    return {handshake.name = "return0"}
  }
}

