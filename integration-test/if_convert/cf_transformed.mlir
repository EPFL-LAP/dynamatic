module {
  func.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>) {
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c10000_i32 = arith.constant {handshake.name = "constant1"} 10000 : i32
    %c2_i32 = arith.constant {handshake.name = "constant2"} 2 : i32
    %c199_i32 = arith.constant {handshake.name = "constant3"} 199 : i32
    cf.br ^bb4(%c1_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb1
    %1 = arith.extsi %0 {handshake.name = "extsi0"} : i32 to i64
    %2 = arith.index_cast %1 {handshake.name = "index_cast0"} : i64 to index
    %3 = memref.load %arg0[%2] {handshake.name = "load0"} : memref<200xi32>
    %4 = arith.extsi %0 {handshake.name = "extsi1"} : i32 to i64
    %5 = arith.index_cast %4 {handshake.name = "index_cast1"} : i64 to index
    memref.store %c1_i32, %arg1[%5] {handshake.name = "store0"} : memref<200xi32>
    %6 = arith.muli %0, %3 {handshake.name = "muli0"} : i32
    %7 = arith.cmpi sge, %6, %c10000_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %7, ^bb3, ^bb2 {handshake.name = "cond_br1"}
  ^bb3:
    %9 = arith.addi %0, %c1_i32 {handshake.name = "addi1"} : i32
    %13 = arith.cmpi slt, %9, %c199_i32 {handshake.name = "cmpi3"} : i32
    cf.cond_br %13, ^bb1(%9 : i32), ^bb5 {handshake.name = "cond_br2"}
  ^bb2:
    %8 = arith.addi %0, %c2_i32 {handshake.name = "addi0"} : i32
    %12 = arith.cmpi slt, %8, %c199_i32 {handshake.name = "cmpi2"} : i32
    cf.cond_br %12, ^bb4(%8 : i32), ^bb5 {handshake.name = "cond_br0"}
  ^bb4(%14: i32):
    cf.br ^bb1(%14 : i32) {handshake.name = "br1"}
  ^bb5:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

