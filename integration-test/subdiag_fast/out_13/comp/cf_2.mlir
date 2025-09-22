module {
  func.func @subdiag_fast(%arg0: memref<1000xf32> {handshake.arg_name = "d1"}, %arg1: memref<1000xf32> {handshake.arg_name = "d2"}, %arg2: memref<1000xf32> {handshake.arg_name = "e"}) -> i32 {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %cst = arith.constant {handshake.name = "constant1"} 1.000000e-03 : f32
    %c1_i32 = arith.constant {handshake.name = "constant2"} 1 : i32
    %c999_i32 = arith.constant {handshake.name = "constant3"} 999 : i32
    cf.br ^bb1(%c0_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = arith.extui %0 {handshake.name = "extui0"} : i32 to i64
    %2 = arith.index_cast %1 {handshake.name = "index_cast0"} : i64 to index
    %3 = memref.load %arg0[%2] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : memref<1000xf32>
    %4 = arith.extui %0 {handshake.name = "extui1"} : i32 to i64
    %5 = arith.index_cast %4 {handshake.name = "index_cast1"} : i64 to index
    %6 = memref.load %arg1[%5] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : memref<1000xf32>
    %7 = arith.addf %3, %6 {handshake.name = "addf0"} : f32
    %8 = arith.extui %0 {handshake.name = "extui2"} : i32 to i64
    %9 = arith.index_cast %8 {handshake.name = "index_cast2"} : i64 to index
    %10 = memref.load %arg2[%9] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : memref<1000xf32>
    %11 = arith.mulf %7, %cst {handshake.name = "mulf0"} : f32
    %12 = arith.cmpf ugt, %10, %11 {handshake.name = "cmpf0"} : f32
    cf.cond_br %12, ^bb2, ^bb3(%0 : i32) {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %13 = arith.addi %0, %c1_i32 {handshake.name = "addi0"} : i32
    %14 = arith.cmpi ult, %13, %c999_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %14, ^bb1(%13 : i32), ^bb3(%13 : i32) {handshake.name = "cond_br1"}
  ^bb3(%15: i32):  // 2 preds: ^bb1, ^bb2
    return {handshake.name = "return0"} %15 : i32
  }
}

