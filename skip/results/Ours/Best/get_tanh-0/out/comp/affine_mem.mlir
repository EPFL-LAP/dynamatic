module {
  func.func @get_tanh(%arg0: memref<1000xf32> {handshake.arg_name = "A"}, %arg1: memref<1000xi32> {handshake.arg_name = "addr"}) {
    %cst = arith.constant {handshake.name = "constant0"} 3.70476198 : f32
    %cst_0 = arith.constant {handshake.name = "constant1"} 19.5238094 : f32
    %cst_1 = arith.constant {handshake.name = "constant2"} 1.000000e+00 : f32
    affine.for %arg2 = 0 to 1000 {
      %0 = affine.load %arg1[%arg2] {handshake.name = "load0"} : memref<1000xi32>
      %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : i32 to index
      %2 = memref.load %arg0[%1] {handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.name = "load1"} : memref<1000xf32>
      %3 = arith.cmpf oge, %2, %cst_1 {handshake.name = "cmpf0"} : f32
      %4 = scf.if %3 -> (f32) {
        scf.yield {handshake.name = "yield0"} %cst_1 : f32
      } else {
        %5 = arith.mulf %2, %2 {handshake.name = "mulf0"} : f32
        %6 = arith.addf %5, %cst_0 {handshake.name = "addf0"} : f32
        %7 = arith.mulf %6, %2 {handshake.name = "mulf1"} : f32
        %8 = arith.mulf %7, %2 {handshake.name = "mulf2"} : f32
        %9 = arith.addf %8, %cst {handshake.name = "addf1"} : f32
        %10 = arith.mulf %9, %2 {handshake.name = "mulf3"} : f32
        scf.yield {handshake.name = "yield1"} %10 : f32
      } {handshake.name = "if0"}
      memref.store %4, %arg0[%1] {handshake.deps = #handshake<deps[["load1", 0, true], ["store0", 0, true]]>, handshake.name = "store0"} : memref<1000xf32>
    } {handshake.name = "for0"}
    return {handshake.name = "return0"}
  }
}

