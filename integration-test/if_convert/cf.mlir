module {
  func.func @if_convert(%arg0: memref<200xi32> {handshake.arg_name = "a"}, %arg1: memref<200xi32> {handshake.arg_name = "b"}) {
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c10000_i32 = arith.constant {handshake.name = "constant1"} 10000 : i32
    %c199_i32 = arith.constant {handshake.name = "constant2"} 199 : i32
    cf.br ^bb1(%c1_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : i32 to index
    %2 = memref.load %arg0[%1] {handshake.name = "load0"} : memref<200xi32>
    %3 = arith.muli %0, %2 {handshake.name = "muli0"} : i32
    %4 = arith.cmpi slt, %3, %c10000_i32 {handshake.name = "cmpi0"} : i32
    %5 = arith.addi %0, %c1_i32 {handshake.name = "addi3"} : i32
    %6 = arith.select %4, %5, %0 {handshake.name = "select0"} : i32
    %7 = arith.addi %6, %c1_i32 {handshake.name = "addi1"} : i32
    %8 = arith.index_cast %7 {handshake.name = "index_cast1"} : i32 to index
    memref.store %c1_i32, %arg1[%8] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "store0"} : memref<200xi32>
    %9 = arith.cmpi slt, %7, %c199_i32 {handshake.name = "cmpi1"} : i32
    cf.cond_br %9, ^bb1(%7 : i32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

