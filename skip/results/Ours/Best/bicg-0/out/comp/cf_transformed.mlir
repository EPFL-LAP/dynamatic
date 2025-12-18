module {
  func.func @bicg(%arg0: memref<900xi32> {handshake.arg_name = "a"}, %arg1: memref<30xi32> {handshake.arg_name = "s"}, %arg2: memref<30xi32> {handshake.arg_name = "q"}, %arg3: memref<30xi32> {handshake.arg_name = "p"}, %arg4: memref<30xi32> {handshake.arg_name = "r"}) -> i32 {
    %c1 = arith.constant 1 : index
    %c30 = arith.constant 30 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb3
    %1 = memref.load %arg2[%0] {handshake.deps = #handshake<deps[["store3", 2, true]]>, handshake.name = "load5"} : memref<30xi32>
    cf.br ^bb2(%c0, %1 : index, i32) {handshake.name = "br1"}
  ^bb2(%2: index, %3: i32):  // 2 preds: ^bb1, ^bb2
    %4 = arith.muli %0, %c30 {handshake.name = "muli2"} : index
    %5 = arith.addi %2, %4 {handshake.name = "addi2"} : index
    %6 = memref.load %arg0[%5] {handshake.name = "load10"} : memref<900xi32>
    %7 = memref.load %arg1[%2] {handshake.deps = #handshake<deps[["store2", 1, true], ["store2", 3, true]]>, handshake.name = "load7"} : memref<30xi32>
    %8 = memref.load %arg4[%0] {handshake.name = "load8"} : memref<30xi32>
    %9 = arith.muli %8, %6 {handshake.name = "muli0"} : i32
    %10 = arith.addi %7, %9 {handshake.name = "addi0"} : i32
    memref.store %10, %arg1[%2] {handshake.deps = #handshake<deps[["load7", 1, true], ["store2", 1, true]]>, handshake.name = "store2"} : memref<30xi32>
    %11 = memref.load %arg3[%2] {handshake.name = "load9"} : memref<30xi32>
    %12 = arith.muli %6, %11 {handshake.name = "muli1"} : i32
    %13 = arith.addi %3, %12 {handshake.name = "addi1"} : i32
    %14 = arith.addi %2, %c1 {handshake.name = "addi3"} : index
    %15 = arith.cmpi ult, %14, %c30 {handshake.name = "cmpi0"} : index
    cf.cond_br %15, ^bb2(%14, %13 : index, i32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    memref.store %13, %arg2[%0] {handshake.name = "store3"} : memref<30xi32>
    %16 = arith.addi %0, %c1 {handshake.name = "addi4"} : index
    %17 = arith.cmpi ult, %16, %c30 {handshake.name = "cmpi1"} : index
    cf.cond_br %17, ^bb1(%16 : index), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    return {handshake.name = "return0"} %13 : i32
  }
}

