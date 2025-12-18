module {
  func.func @atax(%arg0: memref<20x20xf32> {handshake.arg_name = "A"}, %arg1: memref<20xf32> {handshake.arg_name = "x"}, %arg2: memref<20xf32> {handshake.arg_name = "y"}, %arg3: memref<20xf32> {handshake.arg_name = "tmp"}) {
    affine.for %arg4 = 0 to 20 {
      %0 = affine.load %arg3[%arg4] {handshake.deps = #handshake<deps[["store1", 2]]>, handshake.name = "load0"} : memref<20xf32>
      %1 = affine.for %arg5 = 0 to 20 iter_args(%arg6 = %0) -> (f32) {
        %2 = affine.load %arg0[%arg4, %arg5] {handshake.name = "load1"} : memref<20x20xf32>
        %3 = affine.load %arg1[%arg5] {handshake.name = "load2"} : memref<20xf32>
        %4 = arith.mulf %2, %3 {handshake.name = "mulf0"} : f32
        %5 = arith.addf %arg6, %4 {handshake.name = "addf0"} : f32
        affine.yield {handshake.name = "yield0"} %5 : f32
      } {handshake.name = "for0"}
      affine.for %arg5 = 0 to 20 {
        %2 = affine.load %arg2[%arg5] {handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.name = "load3"} : memref<20xf32>
        %3 = affine.load %arg0[%arg4, %arg5] {handshake.name = "load4"} : memref<20x20xf32>
        %4 = arith.mulf %3, %1 {handshake.name = "mulf1"} : f32
        %5 = arith.addf %2, %4 {handshake.name = "addf1"} : f32
        affine.store %5, %arg2[%arg5] {handshake.deps = #handshake<deps[["load3", 1], ["store0", 1]]>, handshake.name = "store0"} : memref<20xf32>
      } {handshake.name = "for1"}
      affine.store %1, %arg3[%arg4] {handshake.name = "store1"} : memref<20xf32>
    } {handshake.name = "for2"}
    return {handshake.name = "return0"}
  }
}

