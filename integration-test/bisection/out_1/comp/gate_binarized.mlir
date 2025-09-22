module {
  func.func @bisection(%arg0: f32 {handshake.arg_name = "a"}, %arg1: f32 {handshake.arg_name = "b"}, %arg2: f32 {handshake.arg_name = "tol"}) -> f32 {
    %c0_i32 = arith.constant {handshake.name = "constant4"} 0 : i32
    %cst = arith.constant {handshake.name = "constant5"} -2.000000e+00 : f32
    %0 = arith.mulf %arg0, %arg0 {handshake.name = "mulf0"} : f32
    %1 = arith.addf %0, %cst {handshake.name = "addf0"} : f32
    cf.br ^bb1(%arg0, %arg1, %c0_i32, %1 : f32, f32, i32, f32) {handshake.name = "br0"}
  ^bb1(%30: f32, %31: f32, %32: i32, %33: f32):
    cf.br ^bb2(%30, %31, %32, %33 : f32, f32, i32, f32) {handshake.name = "br2"}
  ^bb2(%2: f32, %3: f32, %4: i32, %5: f32):  // 2 preds: ^bb0, ^bb4
    %cst_0 = arith.constant {handshake.name = "constant6"} -2.000000e+00 : f32
    %cst_1 = arith.constant {handshake.name = "constant7"} 5.000000e-01 : f32
    %6 = arith.addf %2, %3 {handshake.name = "addf1"} : f32
    %7 = arith.mulf %6, %cst_1 {handshake.name = "mulf1"} : f32
    %8 = arith.mulf %7, %7 {handshake.name = "mulf2"} : f32
    %9 = arith.addf %8, %cst_0 {handshake.name = "addf2"} : f32
    %10 = math.absf %9 {handshake.name = "absf0"} : f32
    %11 = arith.cmpf olt, %10, %arg2 {handshake.name = "cmpf0"} : f32
    cf.cond_br %11, ^bb7(%7 : f32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    %cst_2 = arith.constant {handshake.name = "constant8"} 5.000000e-01 : f32
    %12 = arith.subf %3, %2 {handshake.name = "subf0"} : f32
    %13 = arith.mulf %12, %cst_2 {handshake.name = "mulf3"} : f32
    %14 = arith.cmpf olt, %13, %arg2 {handshake.name = "cmpf1"} : f32
    cf.cond_br %14, ^bb8(%7 : f32), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    %cst_3 = arith.constant {handshake.name = "constant9"} 0.000000e+00 : f32
    %c1_i32 = arith.constant {handshake.name = "constant10"} 1 : i32
    %c100_i32 = arith.constant {handshake.name = "constant11"} 100 : i32
    %15 = arith.mulf %5, %9 {handshake.name = "mulf4"} : f32
    %16 = arith.cmpf olt, %15, %cst_3 {handshake.name = "cmpf2"} : f32
    %20 = arith.addi %4, %c1_i32 {handshake.name = "addi0"} : i32
    %21 = arith.cmpi ult, %20, %c100_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %16, ^bb5, ^bb6 {handshake.name = "cond_br4"}
  ^bb5:
    cf.cond_br %21, ^bb2(%2, %7, %20, %5 : f32, f32, i32, f32), ^bb7(%cst_3 : f32) {handshake.name = "cond_br2"}
  ^bb6:
    cf.cond_br %21, ^bb1(%7, %3, %20, %9 : f32, f32, i32, f32), ^bb7(%cst_3 : f32) {handshake.name = "cond_br3"}
  ^bb7(%22: f32):  // 2 preds: ^bb2, ^bb4
    cf.br ^bb8(%22 : f32) {handshake.name = "br1"}
  ^bb8(%23: f32):  // 2 preds: ^bb3, ^bb7
    return {handshake.name = "return0"} %23 : f32
  }
}

