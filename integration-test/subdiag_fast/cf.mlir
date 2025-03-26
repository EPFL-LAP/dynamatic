module {
  func.func @subdiag_fast(%arg0: memref<1000xf32> {handshake.arg_name = "d1"}, %arg1: memref<1000xf32> {handshake.arg_name = "d2"}, %arg2: memref<1000xf32> {handshake.arg_name = "e"}) -> i32 {
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %cst = arith.constant {handshake.name = "constant1"} 1.000000e-03 : f32
    %c0_i8 = arith.constant {handshake.name = "constant2"} 0 : i8
    %c999_i32 = arith.constant {handshake.name = "constant3"} 999 : i32
    %c0_i32 = arith.constant {handshake.name = "constant4"} 0 : i32
    cf.br ^bb1(%c0_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : i32 to index
    %2 = memref.load %arg0[%1] {handshake.name = "load0"} : memref<1000xf32>
    %3 = arith.addi %0, %c1_i32 {handshake.name = "addi0"} : i32
    %4 = arith.index_cast %3 {handshake.name = "index_cast1"} : i32 to index
    %5 = memref.load %arg1[%4] {handshake.name = "load1"} : memref<1000xf32>
    %6 = arith.addf %2, %5 {handshake.name = "addf0"} : f32
    %7 = memref.load %arg2[%4] {handshake.name = "load2"} : memref<1000xf32>
    %8 = arith.mulf %6, %cst {handshake.name = "mulf0"} : f32
    %9 = arith.cmpf ole, %7, %8 {handshake.name = "cmpf0"} : f32
    %10 = arith.extui %9 {handshake.name = "extui0"} : i1 to i8
    %11 = arith.cmpi eq, %10, %c0_i8 {handshake.name = "cmpi0"} : i8
    %12 = arith.cmpi slt, %3, %c999_i32 {handshake.name = "cmpi1"} : i32
    %13 = arith.andi %12, %11 {handshake.name = "andi0"} : i1
    cf.cond_br %13, ^bb1(%3 : i32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"} %3 : i32
  }
}

