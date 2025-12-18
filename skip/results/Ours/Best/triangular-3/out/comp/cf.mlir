module {
  func.func @triangular(%arg0: memref<10xi32> {handshake.arg_name = "x"}, %arg1: i32 {handshake.arg_name = "n"}, %arg2: memref<100xi32> {handshake.arg_name = "a"}) {
    %c10 = arith.constant {handshake.name = "constant19"} 10 : index
    %c-2 = arith.constant {handshake.name = "constant8"} -2 : index
    %c-1 = arith.constant {handshake.name = "constant3"} -1 : index
    %c1 = arith.constant {handshake.name = "constant1"} 1 : index
    %c0 = arith.constant {handshake.name = "constant0"} 0 : index
    %0 = arith.index_cast %arg1 {handshake.name = "index_cast0"} : i32 to index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb5
    %2 = arith.cmpi slt, %1, %0 {handshake.name = "cmpi0"} : index
    cf.cond_br %2, ^bb2, ^bb6 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %3 = arith.muli %1, %c-1 {handshake.name = "muli1"} : index
    %4 = arith.addi %3, %0 {handshake.name = "addi0"} : index
    %5 = arith.addi %4, %c-1 {handshake.name = "addi1"} : index
    cf.br ^bb3(%c0 : index) {handshake.name = "br1"}
  ^bb3(%6: index):  // 2 preds: ^bb2, ^bb4
    %7 = arith.cmpi slt, %6, %5 {handshake.name = "cmpi1"} : index
    cf.cond_br %7, ^bb4, ^bb5 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    %8 = arith.muli %1, %c-1 {handshake.name = "muli2"} : index
    %9 = arith.muli %6, %c-1 {handshake.name = "muli3"} : index
    %10 = arith.addi %8, %9 {handshake.name = "addi2"} : index
    %11 = arith.addi %10, %0 {handshake.name = "addi3"} : index
    %12 = arith.addi %11, %c-2 {handshake.name = "addi4"} : index
    %13 = arith.muli %1, %c-1 {handshake.name = "muli4"} : index
    %14 = arith.addi %13, %0 {handshake.name = "addi5"} : index
    %15 = arith.addi %14, %c-1 {handshake.name = "addi6"} : index
    %16 = arith.muli %12, %c10 {handshake.name = "muli10"} : index
    %17 = arith.addi %15, %16 {handshake.name = "addi15"} : index
    %18 = memref.load %arg2[%17] {handshake.name = "load6"} : memref<100xi32>
    %19 = arith.muli %1, %c-1 {handshake.name = "muli5"} : index
    %20 = arith.addi %19, %0 {handshake.name = "addi7"} : index
    %21 = arith.addi %20, %c-1 {handshake.name = "addi8"} : index
    %22 = memref.load %arg0[%21] {handshake.name = "load4"} : memref<10xi32>
    %23 = arith.muli %18, %22 {handshake.name = "muli0"} : i32
    %24 = arith.muli %1, %c-1 {handshake.name = "muli6"} : index
    %25 = arith.muli %6, %c-1 {handshake.name = "muli7"} : index
    %26 = arith.addi %24, %25 {handshake.name = "addi9"} : index
    %27 = arith.addi %26, %0 {handshake.name = "addi10"} : index
    %28 = arith.addi %27, %c-2 {handshake.name = "addi11"} : index
    %29 = arith.muli %28, %c10 {handshake.name = "muli11"} : index
    %30 = arith.addi %0, %29 {handshake.name = "addi16"} : index
    %31 = memref.load %arg2[%30] {handshake.deps = #handshake<deps[["store2", 1, true], ["store2", 3, true]]>, handshake.name = "load7"} : memref<100xi32>
    %32 = arith.subi %31, %23 {handshake.name = "subi0"} : i32
    %33 = arith.muli %1, %c-1 {handshake.name = "muli8"} : index
    %34 = arith.muli %6, %c-1 {handshake.name = "muli9"} : index
    %35 = arith.addi %33, %34 {handshake.name = "addi12"} : index
    %36 = arith.addi %35, %0 {handshake.name = "addi13"} : index
    %37 = arith.addi %36, %c-2 {handshake.name = "addi14"} : index
    %38 = arith.muli %37, %c10 {handshake.name = "muli12"} : index
    %39 = arith.addi %0, %38 {handshake.name = "addi17"} : index
    memref.store %32, %arg2[%39] {handshake.deps = #handshake<deps[["load7", 1, true], ["store2", 1, true]]>, handshake.name = "store2"} : memref<100xi32>
    %40 = arith.addi %6, %c1 {handshake.name = "addi18"} : index
    cf.br ^bb3(%40 : index) {handshake.name = "br2"}
  ^bb5:  // pred: ^bb3
    %41 = arith.addi %1, %c1 {handshake.name = "addi19"} : index
    cf.br ^bb1(%41 : index) {handshake.name = "br3"}
  ^bb6:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

