module {
  func.func @if_float2(%arg0: f32 {handshake.arg_name = "x0"}, %arg1: memref<100xf32> {handshake.arg_name = "a"}, %arg2: memref<100xf32> {handshake.arg_name = "minus_trace"}) -> f32 {
    %c0_i32 = arith.constant {handshake.name = "constant6"} 0 : i32
    cf.br ^bb1(%c0_i32, %arg0 : i32, f32) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: f32):  // 2 preds: ^bb0, ^bb4
    %cst = arith.constant {handshake.name = "constant7"} -0.899999976 : f32
    %cst_0 = arith.constant {handshake.name = "constant8"} 0.000000e+00 : f32
    %2 = arith.extui %0 {handshake.name = "extui0"} : i32 to i64
    %3 = arith.index_cast %2 {handshake.name = "index_cast0"} : i64 to index
    %4 = memref.load %arg1[%3] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : memref<100xf32>
    %5 = arith.mulf %4, %1 {handshake.name = "mulf0"} : f32
    %6 = arith.mulf %1, %cst {handshake.name = "mulf1"} : f32
    %7 = arith.addf %5, %6 {handshake.name = "addf0"} : f32
    %8 = arith.cmpf ugt, %7, %cst_0 {handshake.name = "cmpf0"} : f32
    cf.cond_br %8, ^bb3, ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %cst_1 = arith.constant {handshake.name = "constant9"} 3.000000e+00 : f32
    %9 = arith.addf %1, %cst_1 {handshake.name = "addf1"} : f32
    cf.br ^bb4(%9 : f32) {handshake.name = "br1"}
  ^bb3:  // pred: ^bb1
    %cst_2 = arith.constant {handshake.name = "constant10"} 1.000000e+00 : f32
    %10 = arith.extui %0 {handshake.name = "extui1"} : i32 to i64
    %11 = arith.index_cast %10 {handshake.name = "index_cast1"} : i64 to index
    memref.store %1, %arg2[%11] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : memref<100xf32>
    %12 = arith.addf %1, %cst_2 {handshake.name = "addf2"} : f32
    cf.br ^bb4(%12 : f32) {handshake.name = "br2"}
  ^bb4(%13: f32):  // 2 preds: ^bb2, ^bb3
    %cst_3 = arith.constant {handshake.name = "constant11"} 1.000000e+00 : f32
    %c1_i32 = arith.constant {handshake.name = "constant12"} 1 : i32
    %c100_i32 = arith.constant {handshake.name = "constant13"} 100 : i32
    %14 = arith.divf %cst_3, %13 {handshake.name = "divf0"} : f32
    %15 = arith.addi %0, %c1_i32 {handshake.name = "addi0"} : i32
    %16 = arith.cmpi ult, %15, %c100_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %16, ^bb1(%15, %14 : i32, f32), ^bb5(%14 : f32) {handshake.name = "cond_br1"}
  ^bb5(%17: f32):  // pred: ^bb4
    return {handshake.name = "return0"} %17 : f32
  }
}

