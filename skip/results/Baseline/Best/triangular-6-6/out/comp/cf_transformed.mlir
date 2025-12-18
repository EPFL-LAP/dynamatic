module {
  func.func @triangular(%arg0: memref<10xi32> {handshake.arg_name = "x"}, %arg1: i32 {handshake.arg_name = "n"}, %arg2: memref<100xi32> {handshake.arg_name = "a"}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1 = arith.constant -1 : index
    %c-2 = arith.constant -2 : index
    %c10 = arith.constant 10 : index
    %0 = arith.index_cast %arg1 {handshake.name = "index_cast0"} : i32 to index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb5
    %2 = arith.cmpi slt, %1, %0 {handshake.name = "cmpi0"} : index
    cf.cond_br %2, ^bb2, ^bb6 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %3 = arith.subi %0, %1 : index
    %4 = arith.addi %3, %c-1 {handshake.name = "addi1"} : index
    cf.br ^bb3(%c0 : index) {handshake.name = "br1"}
  ^bb3(%5: index):  // 2 preds: ^bb2, ^bb4
    %6 = arith.cmpi slt, %5, %4 {handshake.name = "cmpi1"} : index
    cf.cond_br %6, ^bb4, ^bb5 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    %7 = arith.muli %1, %c-1 {handshake.name = "muli2"} : index
    %8 = arith.subi %7, %5 : index
    %9 = arith.addi %8, %0 {handshake.name = "addi3"} : index
    %10 = arith.addi %9, %c-2 {handshake.name = "addi4"} : index
    %11 = arith.addi %3, %c-1 {handshake.name = "addi6"} : index
    %12 = arith.muli %10, %c10 {handshake.name = "muli10"} : index
    %13 = arith.addi %11, %12 {handshake.name = "addi15"} : index
    %14 = memref.load %arg2[%13] {handshake.name = "load0"} : memref<100xi32>
    %15 = arith.addi %3, %c-1 {handshake.name = "addi8"} : index
    %16 = memref.load %arg0[%15] {handshake.name = "load1"} : memref<10xi32>
    %17 = arith.muli %14, %16 {handshake.name = "muli0"} : i32
    %18 = arith.muli %1, %c-1 {handshake.name = "muli6"} : index
    %19 = arith.subi %18, %5 : index
    %20 = arith.addi %19, %0 {handshake.name = "addi10"} : index
    %21 = arith.addi %20, %c-2 {handshake.name = "addi11"} : index
    %22 = arith.muli %21, %c10 {handshake.name = "muli11"} : index
    %23 = arith.addi %0, %22 {handshake.name = "addi16"} : index
    %24 = memref.load %arg2[%23] {handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.name = "load2"} : memref<100xi32>
    %25 = arith.subi %24, %17 {handshake.name = "subi0"} : i32
    %26 = arith.muli %1, %c-1 {handshake.name = "muli8"} : index
    %27 = arith.subi %26, %5 : index
    %28 = arith.addi %27, %0 {handshake.name = "addi13"} : index
    %29 = arith.addi %28, %c-2 {handshake.name = "addi14"} : index
    %30 = arith.muli %29, %c10 {handshake.name = "muli12"} : index
    %31 = arith.addi %0, %30 {handshake.name = "addi17"} : index
    memref.store %25, %arg2[%31] {handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.name = "store0"} : memref<100xi32>
    %32 = arith.addi %5, %c1 {handshake.name = "addi18"} : index
    cf.br ^bb3(%32 : index) {handshake.name = "br2"}
  ^bb5:  // pred: ^bb3
    %33 = arith.addi %1, %c1 {handshake.name = "addi19"} : index
    cf.br ^bb1(%33 : index) {handshake.name = "br3"}
  ^bb6:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

