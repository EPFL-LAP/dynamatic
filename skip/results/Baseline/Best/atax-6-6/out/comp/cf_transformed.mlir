module {
  func.func @atax(%arg0: memref<400xf32> {handshake.arg_name = "A"}, %arg1: memref<20xf32> {handshake.arg_name = "x"}, %arg2: memref<20xf32> {handshake.arg_name = "y"}, %arg3: memref<20xf32> {handshake.arg_name = "tmp"}) {
    %c1 = arith.constant 1 : index
    %c20 = arith.constant 20 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb4
    %1 = memref.load %arg3[%0] {handshake.deps = #handshake<deps[["store1", 2]]>, handshake.name = "load0"} : memref<20xf32>
    cf.br ^bb2(%c0, %1 : index, f32) {handshake.name = "br1"}
  ^bb2(%2: index, %3: f32):  // 2 preds: ^bb1, ^bb2
    %4 = arith.muli %0, %c20 {handshake.name = "muli0"} : index
    %5 = arith.addi %2, %4 {handshake.name = "addi0"} : index
    %6 = memref.load %arg0[%5] {handshake.name = "load1"} : memref<400xf32>
    %7 = memref.load %arg1[%2] {handshake.name = "load2"} : memref<20xf32>
    %8 = arith.mulf %6, %7 {handshake.name = "mulf0"} : f32
    %9 = arith.addf %3, %8 {handshake.name = "addf0"} : f32
    %10 = arith.addi %2, %c1 {handshake.name = "addi2"} : index
    %11 = arith.cmpi ult, %10, %c20 {handshake.name = "cmpi0"} : index
    cf.cond_br %11, ^bb2(%10, %9 : index, f32), ^bb3(%c0 : index)
  ^bb3(%12: index):  // 2 preds: ^bb2, ^bb3
    %13 = memref.load %arg2[%12] {handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.name = "load3"} : memref<20xf32>
    %14 = arith.muli %0, %c20 {handshake.name = "muli1"} : index
    %15 = arith.addi %12, %14 {handshake.name = "addi1"} : index
    %16 = memref.load %arg0[%15] {handshake.name = "load4"} : memref<400xf32>
    %17 = arith.mulf %16, %9 {handshake.name = "mulf1"} : f32
    %18 = arith.addf %13, %17 {handshake.name = "addf1"} : f32
    memref.store %18, %arg2[%12] {handshake.deps = #handshake<deps[["load3", 1], ["store0", 1]]>, handshake.name = "store0"} : memref<20xf32>
    %19 = arith.addi %12, %c1 {handshake.name = "addi3"} : index
    %20 = arith.cmpi ult, %19, %c20 {handshake.name = "cmpi1"} : index
    cf.cond_br %20, ^bb3(%19 : index), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    memref.store %9, %arg3[%0] {handshake.name = "store1"} : memref<20xf32>
    %21 = arith.addi %0, %c1 {handshake.name = "addi4"} : index
    %22 = arith.cmpi ult, %21, %c20 {handshake.name = "cmpi2"} : index
    cf.cond_br %22, ^bb1(%21 : index), ^bb5 {handshake.name = "cond_br2"}
  ^bb5:  // pred: ^bb4
    return {handshake.name = "return0"}
  }
}

