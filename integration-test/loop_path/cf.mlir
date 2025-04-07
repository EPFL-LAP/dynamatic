module {
  func.func @loop_path(%arg0: memref<1000xi32> {handshake.arg_name = "a"}, %arg1: memref<1000xi32> {handshake.arg_name = "b"}, %arg2: memref<1000xi32> {handshake.arg_name = "c"}) {
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c5_i32 = arith.constant {handshake.name = "constant1"} 5 : i32
    %c0_i8 = arith.constant {handshake.name = "constant2"} 0 : i8
    %c1000_i32 = arith.constant {handshake.name = "constant3"} 1000 : i32
    %c0_i32 = arith.constant {handshake.name = "constant4"} 0 : i32
    cf.br ^bb1(%c0_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : i32 to index
    %2 = memref.load %arg0[%1] {handshake.name = "load0"} : memref<1000xi32>
    %3 = memref.load %arg1[%1] {handshake.name = "load1"} : memref<1000xi32>
    %4 = arith.addi %2, %3 {handshake.name = "addi0"} : i32
    memref.store %4, %arg2[%1] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "store0"} : memref<1000xi32>
    %5 = arith.addi %0, %c1_i32 {handshake.name = "addi1"} : i32
    %6 = arith.subi %c1000_i32, %4 {handshake.name = "subi0"} : i32
    %7 = arith.muli %4, %c5_i32 {handshake.name = "muli0"} : i32
    %8 = arith.cmpi sle, %6, %7 {handshake.name = "cmpi0"} : i32
    %9 = arith.extui %8 {handshake.name = "extui0"} : i1 to i8
    %10 = arith.cmpi eq, %9, %c0_i8 {handshake.name = "cmpi1"} : i8
    %11 = arith.cmpi slt, %5, %c1000_i32 {handshake.name = "cmpi2"} : i32
    %12 = arith.andi %11, %10 {handshake.name = "andi0"} : i1
    cf.cond_br %12, ^bb1(%5 : i32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

