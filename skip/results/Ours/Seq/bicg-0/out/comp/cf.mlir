module {
  func.func @bicg(%arg0: memref<900xi32> {handshake.arg_name = "a"}, %arg1: memref<30xi32> {handshake.arg_name = "s"}, %arg2: memref<30xi32> {handshake.arg_name = "q"}, %arg3: memref<30xi32> {handshake.arg_name = "p"}, %arg4: memref<30xi32> {handshake.arg_name = "r"}) -> i32 {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %c30 = arith.constant {handshake.name = "constant2"} 30 : index
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    cf.br ^bb1(%c0, %c0_i32 : index, i32) {handshake.name = "br0"}
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb3
    %2 = memref.load %arg2[%0] {handshake.deps = #handshake<deps[["store3", 2, true]]>, handshake.name = "load5"} : memref<30xi32>
    cf.br ^bb2(%c0, %2 : index, i32) {handshake.name = "br1"}
  ^bb2(%3: index, %4: i32):  // 2 preds: ^bb1, ^bb2
    %5 = arith.muli %0, %c30 {handshake.name = "muli2"} : index
    %6 = arith.addi %3, %5 {handshake.name = "addi2"} : index
    %7 = memref.load %arg0[%6] {handshake.name = "load10"} : memref<900xi32>
    %8 = memref.load %arg1[%3] {handshake.deps = #handshake<deps[["store2", 1, true], ["store2", 3, true]]>, handshake.name = "load7"} : memref<30xi32>
    %9 = memref.load %arg4[%0] {handshake.name = "load8"} : memref<30xi32>
    %10 = arith.muli %9, %7 {handshake.name = "muli0"} : i32
    %11 = arith.addi %8, %10 {handshake.name = "addi0"} : i32
    memref.store %11, %arg1[%3] {handshake.deps = #handshake<deps[["load7", 1, true], ["store2", 1, true]]>, handshake.name = "store2"} : memref<30xi32>
    %12 = memref.load %arg3[%3] {handshake.name = "load9"} : memref<30xi32>
    %13 = arith.muli %7, %12 {handshake.name = "muli1"} : i32
    %14 = arith.addi %4, %13 {handshake.name = "addi1"} : i32
    %15 = arith.addi %3, %c1 {handshake.name = "addi3"} : index
    %16 = arith.cmpi ult, %15, %c30 {handshake.name = "cmpi0"} : index
    cf.cond_br %16, ^bb2(%15, %14 : index, i32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    memref.store %14, %arg2[%0] {handshake.name = "store3"} : memref<30xi32>
    %17 = arith.addi %0, %c1 {handshake.name = "addi4"} : index
    %18 = arith.cmpi ult, %17, %c30 {handshake.name = "cmpi1"} : index
    cf.cond_br %18, ^bb1(%17, %14 : index, i32), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    return {handshake.name = "return0"} %14 : i32
  }
}

