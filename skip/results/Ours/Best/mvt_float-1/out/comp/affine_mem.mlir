module {
  func.func @mvt_float(%arg0: memref<30x30xf32> {handshake.arg_name = "A"}, %arg1: memref<30xf32> {handshake.arg_name = "x1"}, %arg2: memref<30xf32> {handshake.arg_name = "x2"}, %arg3: memref<30xf32> {handshake.arg_name = "y1"}, %arg4: memref<30xf32> {handshake.arg_name = "y2"}) {
    affine.for %arg5 = 0 to 30 {
      %0 = affine.load %arg1[%arg5] {handshake.deps = #handshake<deps[["store0", 2, true]]>, handshake.name = "load0"} : memref<30xf32>
      %1 = affine.for %arg6 = 0 to 30 iter_args(%arg7 = %0) -> (f32) {
        %2 = affine.load %arg0[%arg5, %arg6] {handshake.name = "load1"} : memref<30x30xf32>
        %3 = affine.load %arg3[%arg6] {handshake.name = "load2"} : memref<30xf32>
        %4 = arith.mulf %2, %3 {handshake.name = "mulf0"} : f32
        %5 = arith.addf %arg7, %4 {handshake.name = "addf0"} : f32
        affine.yield {handshake.name = "yield0"} %5 : f32
      } {handshake.name = "for0"}
      affine.store %1, %arg1[%arg5] {handshake.name = "store0"} : memref<30xf32>
    } {handshake.name = "for1"}
    affine.for %arg5 = 0 to 30 {
      %0 = affine.load %arg2[%arg5] {handshake.deps = #handshake<deps[["store1", 2, true]]>, handshake.name = "load3"} : memref<30xf32>
      %1 = affine.for %arg6 = 0 to 30 iter_args(%arg7 = %0) -> (f32) {
        %2 = affine.load %arg0[%arg6, %arg5] {handshake.name = "load4"} : memref<30x30xf32>
        %3 = affine.load %arg4[%arg6] {handshake.name = "load5"} : memref<30xf32>
        %4 = arith.mulf %2, %3 {handshake.name = "mulf1"} : f32
        %5 = arith.addf %arg7, %4 {handshake.name = "addf1"} : f32
        affine.yield {handshake.name = "yield2"} %5 : f32
      } {handshake.name = "for2"}
      affine.store %1, %arg2[%arg5] {handshake.name = "store1"} : memref<30xf32>
    } {handshake.name = "for3"}
    return {handshake.name = "return0"}
  }
}

