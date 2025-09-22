module {
  func.func @bisection(%arg0: f32 {handshake.arg_name = "a"}, %arg1: f32 {handshake.arg_name = "b"}, %arg2: f32 {handshake.arg_name = "tol"}) -> f32 {
    %cst = arith.constant {handshake.name = "constant0"} -2.000000e+00 : f32
    %c0_i32 = arith.constant {handshake.name = "constant1"} 0 : i32
    %cst_0 = arith.constant {handshake.name = "constant2"} 5.000000e-01 : f32
    %cst_1 = arith.constant {handshake.name = "constant3"} 0.000000e+00 : f32
    %c1_i32 = arith.constant {handshake.name = "constant4"} 1 : i32
    %c100_i32 = arith.constant {handshake.name = "constant5"} 100 : i32
    %0 = arith.mulf %arg0, %arg0 {handshake.name = "mulf0"} : f32
    %1 = arith.addf %0, %cst {handshake.name = "addf0"} : f32
    cf.br ^bb1(%arg0, %arg1, %c0_i32, %1 : f32, f32, i32, f32) {handshake.name = "br0"}
  ^bb1(%2: f32, %3: f32, %4: i32, %5: f32):  // 2 preds: ^bb0, ^bb3
    %6 = arith.addf %2, %3 {handshake.name = "addf1"} : f32
    %7 = arith.mulf %6, %cst_0 {handshake.name = "mulf1"} : f32
    %8 = arith.mulf %7, %7 {handshake.name = "mulf2"} : f32
    %9 = arith.addf %8, %cst {handshake.name = "addf2"} : f32
    %10 = math.absf %9 {handshake.name = "absf0"} : f32
    %11 = arith.cmpf olt, %10, %arg2 {handshake.name = "cmpf0"} : f32
    cf.cond_br %11, ^bb4(%7 : f32), ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %12 = arith.subf %3, %2 {handshake.name = "subf0"} : f32
    %13 = arith.mulf %12, %cst_0 {handshake.name = "mulf3"} : f32
    %14 = arith.cmpf olt, %13, %arg2 {handshake.name = "cmpf1"} : f32
    cf.cond_br %14, ^bb4(%7 : f32), ^bb3 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %15 = arith.mulf %5, %9 {handshake.name = "mulf4"} : f32
    %16 = arith.cmpf olt, %15, %cst_1 {handshake.name = "cmpf2"} : f32
    %17 = arith.select %16, %7, %3 {handshake.name = "select0"} : f32
    %18 = arith.select %16, %2, %7 {handshake.name = "select1"} : f32
    %19 = arith.select %16, %5, %9 {handshake.name = "select2"} : f32
    %20 = arith.addi %4, %c1_i32 {handshake.name = "addi0"} : i32
    %21 = arith.cmpi ult, %20, %c100_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %21, ^bb1(%18, %17, %20, %19 : f32, f32, i32, f32), ^bb4(%cst_1 : f32) {handshake.name = "cond_br2"}
  ^bb4(%22: f32):  // 3 preds: ^bb1, ^bb2, ^bb3
    return {handshake.name = "return0"} %22 : f32
  }
}

