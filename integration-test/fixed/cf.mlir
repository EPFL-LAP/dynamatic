module {
  func.func @fixed(%arg0: f32 {handshake.arg_name = "y"}) -> f32 {
    %cst = arith.constant {handshake.name = "constant0"} 9.99999993E-9 : f32
    %cst_0 = arith.constant {handshake.name = "constant1"} 1.000000e+00 : f32
    cf.br ^bb1(%cst_0 : f32) {handshake.name = "br0"}
  ^bb1(%0: f32):  // 2 preds: ^bb0, ^bb2
    %1 = arith.mulf %0, %arg0 {handshake.name = "mulf0"} : f32
    %2 = arith.subf %0, %1 {handshake.name = "subf0"} : f32
    %3 = arith.cmpf oge, %2, %cst {handshake.name = "cmpf0"} : f32
    cf.cond_br %3, ^bb1(%1 : f32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"} %1 : f32
  }
}

