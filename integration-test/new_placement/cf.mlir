module {
  func.func @new_placement(%arg0: memref<1000xi32> {handshake.arg_name = "a"}, %arg1: memref<1000xi32> {handshake.arg_name = "b"}, %arg2: memref<1000xi32> {handshake.arg_name = "c"}) {
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c2_i32 = arith.constant {handshake.name = "constant1"} 2 : i32
    %c1000_i32 = arith.constant {handshake.name = "constant2"} 1000 : i32
    %c0_i32 = arith.constant {handshake.name = "constant3"} 0 : i32
    cf.br ^bb1(%c0_i32, %c0_i32 : i32, i32) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: i32):  // 2 preds: ^bb0, ^bb2
    %2 = arith.index_cast %1 {handshake.name = "index_cast0"} : i32 to index
    %3 = memref.load %arg0[%2] {handshake.name = "load0"} : memref<1000xi32>
    %4 = memref.load %arg1[%2] {handshake.name = "load1"} : memref<1000xi32>
    %5 = arith.muli %3, %4 {handshake.name = "muli0"} : i32
    %6 = arith.addi %5, %0 {handshake.name = "addi0"} : i32
    %7 = arith.addi %1, %c1_i32 {handshake.name = "addi1"} : i32
    %8 = arith.muli %7, %c2_i32 {handshake.name = "muli1"} : i32
    %9 = arith.index_cast %7 {handshake.name = "index_cast1"} : i32 to index
    %10 = arith.addi %6, %8 {handshake.name = "addi2"} : i32
    memref.store %10, %arg2[%9] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "store0"} : memref<1000xi32>
    %11 = arith.cmpi slt, %6, %c1000_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %11, ^bb1(%8, %7 : i32, i32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

