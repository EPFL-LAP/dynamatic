module {
  func.func @subdiag_fast(%arg0: memref<1000xf32> {handshake.arg_name = "d1"}, %arg1: memref<1000xf32> {handshake.arg_name = "d2"}, %arg2: memref<1000xf32> {handshake.arg_name = "e"}) -> i32 {
    %c998_i32 = arith.constant 998 : i32
    %cst = arith.constant 1.000000e-03 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb1
    %1 = arith.extui %0 {handshake.name = "extui0"} : i32 to i64
    %2 = arith.index_cast %1 {handshake.name = "index_cast0"} : i64 to index
    %3 = memref.load %arg0[%2] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : memref<1000xf32>
    %4 = arith.addi %0, %c1_i32 {handshake.name = "addi0"} : i32
    %5 = arith.extui %4 {handshake.name = "extui1"} : i32 to i64
    %6 = arith.index_cast %5 {handshake.name = "index_cast1"} : i64 to index
    %7 = memref.load %arg1[%6] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : memref<1000xf32>
    %8 = arith.addf %3, %7 {handshake.name = "addf0"} : f32
    %9 = arith.addi %0, %c1_i32 {handshake.name = "addi1"} : i32
    %10 = arith.extui %0 {handshake.name = "extui2"} : i32 to i64
    %11 = arith.index_cast %10 {handshake.name = "index_cast2"} : i64 to index
    %12 = memref.load %arg2[%11] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : memref<1000xf32>
    %13 = arith.mulf %8, %cst {handshake.name = "mulf0"} : f32
    %14 = arith.cmpf ugt, %12, %13 {handshake.name = "cmpf0"} : f32
    %15 = arith.cmpi ult, %0, %c998_i32 {handshake.name = "cmpi0"} : i32
    %16 = arith.andi %15, %14 : i1
    cf.cond_br %16, ^bb1(%9 : i32), ^bb2(%0 : i32) {handshake.name = "cond_br0"}
  ^bb2(%17: i32):  // pred: ^bb1
    return {handshake.name = "return0"} %17 : i32
  }
}

