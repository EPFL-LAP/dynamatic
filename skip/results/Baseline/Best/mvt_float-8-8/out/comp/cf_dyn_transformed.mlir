module {
  func.func @mvt_float(%arg0: memref<900xf32> {handshake.arg_name = "A"}, %arg1: memref<30xf32> {handshake.arg_name = "x1"}, %arg2: memref<30xf32> {handshake.arg_name = "x2"}, %arg3: memref<30xf32> {handshake.arg_name = "y1"}, %arg4: memref<30xf32> {handshake.arg_name = "y2"}) {
    %c0 = arith.constant {handshake.name = "constant2"} 0 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb3
    %c0_0 = arith.constant {handshake.name = "constant3"} 0 : index
    %1 = memref.load %arg1[%0] {handshake.deps = #handshake<deps[["store0", 2]]>, handshake.name = "load0"} : memref<30xf32>
    cf.br ^bb2(%c0_0, %1 : index, f32) {handshake.name = "br1"}
  ^bb2(%2: index, %3: f32):  // 2 preds: ^bb1, ^bb2
    %c30 = arith.constant {handshake.name = "constant4"} 30 : index
    %c1 = arith.constant {handshake.name = "constant5"} 1 : index
    %4 = arith.muli %0, %c30 {handshake.name = "muli0"} : index
    %5 = arith.addi %2, %4 {handshake.name = "addi0"} : index
    %6 = memref.load %arg0[%5] {handshake.name = "load1"} : memref<900xf32>
    %7 = memref.load %arg3[%2] {handshake.name = "load2"} : memref<30xf32>
    %8 = arith.mulf %6, %7 {handshake.name = "mulf0"} : f32
    %9 = arith.addf %3, %8 {handshake.name = "addf0"} : f32
    %10 = arith.addi %2, %c1 {handshake.name = "addi2"} : index
    %11 = arith.cmpi ult, %10, %c30 {handshake.name = "cmpi0"} : index
    cf.cond_br %11, ^bb2(%10, %9 : index, f32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    %c0_1 = arith.constant {handshake.name = "constant6"} 0 : index
    %c30_2 = arith.constant {handshake.name = "constant7"} 30 : index
    %c1_3 = arith.constant {handshake.name = "constant8"} 1 : index
    memref.store %9, %arg1[%0] {handshake.name = "store0"} : memref<30xf32>
    %12 = arith.addi %0, %c1_3 {handshake.name = "addi3"} : index
    %13 = arith.cmpi ult, %12, %c30_2 {handshake.name = "cmpi1"} : index
    cf.cond_br %13, ^bb1(%12 : index), ^bb4(%c0_1 : index) {handshake.name = "cond_br1"}
  ^bb4(%14: index):  // 2 preds: ^bb3, ^bb6
    %c0_4 = arith.constant {handshake.name = "constant9"} 0 : index
    %15 = memref.load %arg2[%14] {handshake.deps = #handshake<deps[["store1", 2]]>, handshake.name = "load3"} : memref<30xf32>
    cf.br ^bb5(%c0_4, %15 : index, f32) {handshake.name = "br3"}
  ^bb5(%16: index, %17: f32):  // 2 preds: ^bb4, ^bb5
    %c30_5 = arith.constant {handshake.name = "constant10"} 30 : index
    %c1_6 = arith.constant {handshake.name = "constant11"} 1 : index
    %18 = arith.muli %16, %c30_5 {handshake.name = "muli1"} : index
    %19 = arith.addi %14, %18 {handshake.name = "addi1"} : index
    %20 = memref.load %arg0[%19] {handshake.name = "load4"} : memref<900xf32>
    %21 = memref.load %arg4[%16] {handshake.name = "load5"} : memref<30xf32>
    %22 = arith.mulf %20, %21 {handshake.name = "mulf1"} : f32
    %23 = arith.addf %17, %22 {handshake.name = "addf1"} : f32
    %24 = arith.addi %16, %c1_6 {handshake.name = "addi4"} : index
    %25 = arith.cmpi ult, %24, %c30_5 {handshake.name = "cmpi2"} : index
    cf.cond_br %25, ^bb5(%24, %23 : index, f32), ^bb6 {handshake.name = "cond_br2"}
  ^bb6:  // pred: ^bb5
    %c30_7 = arith.constant {handshake.name = "constant12"} 30 : index
    %c1_8 = arith.constant {handshake.name = "constant13"} 1 : index
    memref.store %23, %arg2[%14] {handshake.name = "store1"} : memref<30xf32>
    %26 = arith.addi %14, %c1_8 {handshake.name = "addi5"} : index
    %27 = arith.cmpi ult, %26, %c30_7 {handshake.name = "cmpi3"} : index
    cf.cond_br %27, ^bb4(%26 : index), ^bb7 {handshake.name = "cond_br3"}
  ^bb7:  // pred: ^bb6
    return {handshake.name = "return0"}
  }
}

