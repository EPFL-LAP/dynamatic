module {
  func.func @gaussian(%arg0: memref<20xi32> {handshake.arg_name = "c"}, %arg1: memref<400xi32> {handshake.arg_name = "a"}) -> i32 {
    %c20 = arith.constant {handshake.name = "constant9"} 20 : index
    %c19 = arith.constant {handshake.name = "constant3"} 19 : index
    %c1 = arith.constant {handshake.name = "constant2"} 1 : index
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant1"} 0 : i32
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    cf.br ^bb1(%c1, %c0_i32 : index, i32) {handshake.name = "br0"}
  ^bb1(%1: index, %2: i32):  // 2 preds: ^bb0, ^bb6
    %3 = arith.addi %1, %c1 {handshake.name = "addi2"} : index
    cf.br ^bb2(%3, %2 : index, i32) {handshake.name = "br1"}
  ^bb2(%4: index, %5: i32):  // 2 preds: ^bb1, ^bb5
    %6 = arith.cmpi ult, %4, %c19 {handshake.name = "cmpi2"} : index
    cf.cond_br %6, ^bb3, ^bb6 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    cf.br ^bb4(%c1, %c1_i32, %5, %5, %0, %c1_i32, %5 : index, i32, i32, i32, i32, i32, i32) {handshake.name = "br2"}
  ^bb4(%7: index, %8: i32, %9: i32, %10: i32, %11: i32, %12: i32, %13: i32):  // 2 preds: ^bb3, ^bb4
    %14 = arith.index_cast %8 {handshake.name = "index_cast0"} : i32 to index
    %15 = arith.muli %4, %c20 {handshake.name = "muli1"} : index
    %16 = arith.addi %14, %15 {handshake.name = "addi3"} : index
    %17 = memref.load %arg1[%16] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.name = "load4"} : memref<400xi32>
    %18 = memref.load %arg0[%1] {handshake.name = "load3"} : memref<20xi32>
    %19 = arith.muli %1, %c20 {handshake.name = "muli2"} : index
    %20 = arith.addi %14, %19 {handshake.name = "addi4"} : index
    %21 = memref.load %arg1[%20] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.name = "load5"} : memref<400xi32>
    %22 = arith.muli %18, %21 {handshake.name = "muli0"} : i32
    %23 = arith.subi %17, %22 {handshake.name = "subi0"} : i32
    %24 = arith.muli %4, %c20 {handshake.name = "muli3"} : index
    %25 = arith.addi %14, %24 {handshake.name = "addi5"} : index
    memref.store %23, %arg1[%25] {handshake.deps = #handshake<deps[["load4", 0, true], ["load5", 0, true], ["store1", 0, true]]>, handshake.name = "store1"} : memref<400xi32>
    %26 = arith.addi %9, %8 {handshake.name = "addi0"} : i32
    %27 = arith.addi %8, %c1_i32 {handshake.name = "addi1"} : i32
    %28 = arith.addi %7, %c1 {handshake.name = "addi6"} : index
    %29 = arith.cmpi ult, %28, %c20 {handshake.name = "cmpi0"} : index
    cf.cond_br %29, ^bb4(%28, %27, %26, %26, %0, %27, %26 : index, i32, i32, i32, i32, i32, i32), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb4
    %30 = arith.addi %4, %c1 {handshake.name = "addi8"} : index
    cf.br ^bb2(%30, %26 : index, i32) {handshake.name = "br3"}
  ^bb6:  // pred: ^bb2
    %31 = arith.addi %1, %c1 {handshake.name = "addi7"} : index
    %32 = arith.cmpi ult, %31, %c19 {handshake.name = "cmpi1"} : index
    cf.cond_br %32, ^bb1(%31, %5 : index, i32), ^bb7 {handshake.name = "cond_br2"}
  ^bb7:  // pred: ^bb6
    return {handshake.name = "return0"} %5 : i32
  }
}

