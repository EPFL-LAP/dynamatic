module {
  func.func @gaussian(%arg0: memref<20xi32> {handshake.arg_name = "c"}, %arg1: memref<400xi32> {handshake.arg_name = "a"}) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c19 = arith.constant 19 : index
    %c20 = arith.constant 20 : index
    cf.br ^bb1(%c1, %c0_i32 : index, i32) {handshake.name = "br0"}
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb5
    %2 = arith.addi %0, %c1 {handshake.name = "addi2"} : index
    cf.br ^bb2(%2, %1 : index, i32) {handshake.name = "br1"}
  ^bb2(%3: index, %4: i32):  // 2 preds: ^bb1, ^bb4
    %5 = arith.cmpi ult, %3, %c19 {handshake.name = "cmpi2"} : index
    cf.cond_br %5, ^bb3(%c1, %c1_i32, %4 : index, i32, i32), ^bb5
  ^bb3(%6: index, %7: i32, %8: i32):  // 2 preds: ^bb2, ^bb3
    %9 = arith.index_cast %7 {handshake.name = "index_cast0"} : i32 to index
    %10 = arith.muli %3, %c20 {handshake.name = "muli1"} : index
    %11 = arith.addi %9, %10 {handshake.name = "addi3"} : index
    %12 = memref.load %arg1[%11] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load0"} : memref<400xi32>
    %13 = memref.load %arg0[%0] {handshake.name = "load1"} : memref<20xi32>
    %14 = arith.muli %0, %c20 {handshake.name = "muli2"} : index
    %15 = arith.addi %9, %14 {handshake.name = "addi4"} : index
    %16 = memref.load %arg1[%15] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load2"} : memref<400xi32>
    %17 = arith.muli %13, %16 {handshake.name = "muli0"} : i32
    %18 = arith.subi %12, %17 {handshake.name = "subi0"} : i32
    %19 = arith.muli %3, %c20 {handshake.name = "muli3"} : index
    %20 = arith.addi %9, %19 {handshake.name = "addi5"} : index
    memref.store %18, %arg1[%20] {handshake.deps = #handshake<deps[["load0", 0], ["load2", 0], ["store0", 0]]>, handshake.name = "store0"} : memref<400xi32>
    %21 = arith.addi %8, %7 {handshake.name = "addi0"} : i32
    %22 = arith.addi %7, %c1_i32 {handshake.name = "addi1"} : i32
    %23 = arith.addi %6, %c1 {handshake.name = "addi6"} : index
    %24 = arith.cmpi ult, %23, %c20 {handshake.name = "cmpi0"} : index
    cf.cond_br %24, ^bb3(%23, %22, %21 : index, i32, i32), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    %25 = arith.addi %3, %c1 {handshake.name = "addi8"} : index
    cf.br ^bb2(%25, %21 : index, i32) {handshake.name = "br3"}
  ^bb5:  // pred: ^bb2
    %26 = arith.addi %0, %c1 {handshake.name = "addi7"} : index
    %27 = arith.cmpi ult, %26, %c19 {handshake.name = "cmpi1"} : index
    cf.cond_br %27, ^bb1(%26, %4 : index, i32), ^bb6 {handshake.name = "cond_br2"}
  ^bb6:  // pred: ^bb5
    return {handshake.name = "return0"} %4 : i32
  }
}

