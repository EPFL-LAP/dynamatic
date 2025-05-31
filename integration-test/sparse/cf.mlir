module {
  func.func @sparse(%arg0: memref<100xf32> {handshake.arg_name = "a"}, %arg1: memref<100xf32> {handshake.arg_name = "x"}) -> f32 {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %cst = arith.constant {handshake.name = "constant1"} 0.000000e+00 : f32
    %c1_i32 = arith.constant {handshake.name = "constant2"} 1 : i32
    cf.br ^bb1(%c0_i32, %cst : i32, f32) {handshake.name = "br0"}
  ^bb1(%0: i32, %1: f32):  // 2 preds: ^bb0, ^bb2
    %2 = arith.index_cast %0 {handshake.name = "index_cast0"} : i32 to index
    %3 = memref.load %arg0[%2] {handshake.name = "load0"} : memref<100xf32>
    %4 = memref.load %arg1[%2] {handshake.name = "load1"} : memref<100xf32>
    %5 = arith.mulf %3, %4 {handshake.name = "mulf0"} : f32
    %6 = arith.addf %1, %5 {handshake.name = "addf0"} : f32
    %7 = arith.cmpf oge, %6, %cst {handshake.name = "cmpf0"} : f32
    %13 = arith.mulf %3, %4 {handshake.name = "mulf1"} : f32
    %14 = arith.addf %1, %13 {handshake.name = "addf1"} : f32
    %15 = arith.addi %0, %c1_i32 {handshake.name = "addi0"} : i32
    cf.cond_br %7, ^bb1(%15, %14 : i32, f32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"} %6 : f32
  }
}

