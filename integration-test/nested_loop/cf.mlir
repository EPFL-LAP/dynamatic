module {
  func.func @nested_loop(%arg0: memref<1000xi32> {handshake.arg_name = "a"}, %arg1: memref<1000xi32> {handshake.arg_name = "b"}, %arg2: memref<1000xi32> {handshake.arg_name = "c"}) {
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c400_i32 = arith.constant {handshake.name = "constant1"} 400 : i32
    %c1000_i32 = arith.constant {handshake.name = "constant2"} 1000 : i32
    %c0_i32 = arith.constant {handshake.name = "constant3"} 0 : i32
    %c0 = arith.constant {handshake.name = "constant4"} 0 : index
    %c2 = arith.constant {handshake.name = "constant5"} 2 : index
    %c1 = arith.constant {handshake.name = "constant6"} 1 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb4
    %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : index to i32
    %2 = arith.muli %1, %c400_i32 {handshake.name = "muli0"} : i32
    cf.br ^bb2(%c0_i32 : i32) {handshake.name = "br1"}
  ^bb2(%3: i32):  // 2 preds: ^bb1, ^bb3
    %4 = arith.index_cast %3 {handshake.name = "index_cast1"} : i32 to index
    %5 = memref.load %arg0[%4] {handshake.name = "load0"} : memref<1000xi32>
    %6 = memref.load %arg1[%4] {handshake.name = "load1"} : memref<1000xi32>
    %7 = arith.muli %5, %6 {handshake.name = "muli1"} : i32
    %8 = arith.addi %3, %2 {handshake.name = "addi0"} : i32
    %9 = arith.index_cast %8 {handshake.name = "index_cast2"} : i32 to index
    memref.store %7, %arg2[%9] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "store0"} : memref<1000xi32>
    %10 = arith.cmpi slt, %7, %c1000_i32 {handshake.name = "cmpi0"} : i32
    %12 = arith.addi %3, %c1_i32 {handshake.name = "addi1"} : i32
    cf.cond_br %10, ^bb2(%12 : i32), ^bb4 {handshake.name = "cond_br0"}
  ^bb4:  // pred: ^bb2
    %13 = arith.addi %0, %c1 {handshake.name = "addi2"} : index
    %14 = arith.cmpi ult, %13, %c2 {handshake.name = "cmpi1"} : index
    cf.cond_br %14, ^bb1(%13 : index), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb4
    return {handshake.name = "return0"}
  }
}

