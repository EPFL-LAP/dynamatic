module {
  func.func @golden_ratio(%arg0: f32 {handshake.arg_name = "x0"}, %arg1: f32 {handshake.arg_name = "x1"}) -> f32 {
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0_i32, %arg1, %arg0 : i32, f32, f32) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: f32, %2: f32):  // 2 preds: ^bb0, ^bb3
    cf.br ^bb2(%2 : f32) {handshake.name = "br1"}
  ^bb2(%3: f32):  // 2 preds: ^bb1, ^bb2
    %4 = arith.mulf %3, %1 {handshake.name = "mulf0"} : f32
    %5 = arith.addf %3, %4 {handshake.name = "addf0"} : f32
    %6 = arith.mulf %5, %cst_1 {handshake.name = "mulf1"} : f32
    %7 = arith.subf %6, %3 {handshake.name = "subf0"} : f32
    %8 = math.absf %7 {handshake.name = "absf0"} : f32
    %9 = arith.cmpf olt, %8, %cst_0 {handshake.name = "cmpf0"} : f32
    cf.cond_br %9, ^bb3(%3 : f32), ^bb2(%6 : f32) {handshake.name = "cond_br0"}
  ^bb3(%10: f32):  // pred: ^bb2
    %11 = arith.addf %10, %cst {handshake.name = "addf1"} : f32
    %12 = arith.divf %cst, %11 {handshake.name = "divf0"} : f32
    %13 = arith.addi %0, %c1_i32 {handshake.name = "addi0"} : i32
    %14 = arith.cmpi ult, %13, %c100_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %14, ^bb1(%13, %12, %11 : i32, f32, f32), ^bb4(%11 : f32) {handshake.name = "cond_br1"}
  ^bb4(%15: f32):  // pred: ^bb3
    return {handshake.name = "return0"} %15 : f32
  }
}

