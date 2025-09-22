module {
  func.func @golden_ratio(%arg0: f32 {handshake.arg_name = "x0"}) -> f32 {
    %c0_i32 = arith.constant {handshake.name = "constant4"} 0 : i32
    %cst = arith.constant {handshake.name = "constant5"} 1.000000e+00 : f32
    %0 = arith.divf %cst, %arg0 {handshake.name = "divf0"} : f32
    cf.br ^bb1(%0, %c0_i32, %arg0 : f32, i32, f32) {handshake.name = "br0"}
  ^bb1(%1: f32, %2: i32, %3: f32):  // 2 preds: ^bb0, ^bb3
    cf.br ^bb2(%3 : f32) {handshake.name = "br1"}
  ^bb2(%4: f32):  // 2 preds: ^bb1, ^bb2
    %cst_0 = arith.constant {handshake.name = "constant6"} 5.000000e-01 : f32
    %cst_1 = arith.constant {handshake.name = "constant7"} 1.000000e-01 : f32
    %5 = arith.mulf %4, %1 {handshake.name = "mulf0"} : f32
    %6 = arith.addf %4, %5 {handshake.name = "addf0"} : f32
    %7 = arith.mulf %6, %cst_0 {handshake.name = "mulf1"} : f32
    %8 = arith.subf %7, %4 {handshake.name = "subf0"} : f32
    %9 = math.absf %8 {handshake.name = "absf0"} : f32
    %10 = arith.cmpf olt, %9, %cst_1 {handshake.name = "cmpf0"} : f32
    cf.cond_br %10, ^bb3(%4 : f32), ^bb2(%7 : f32) {handshake.name = "cond_br0"}
  ^bb3(%11: f32):  // pred: ^bb2
    %cst_2 = arith.constant {handshake.name = "constant8"} 1.000000e+00 : f32
    %c1_i32 = arith.constant {handshake.name = "constant9"} 1 : i32
    %c100_i32 = arith.constant {handshake.name = "constant10"} 100 : i32
    %12 = arith.addf %11, %cst_2 {handshake.name = "addf1"} : f32
    %13 = arith.addi %2, %c1_i32 {handshake.name = "addi0"} : i32
    %14 = arith.divf %cst_2, %12 {handshake.name = "divf1"} : f32
    %15 = arith.cmpi ult, %13, %c100_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %15, ^bb1(%14, %13, %12 : f32, i32, f32), ^bb4(%12 : f32) {handshake.name = "cond_br1"}
  ^bb4(%16: f32):  // pred: ^bb3
    return {handshake.name = "return0"} %16 : f32
  }
}

