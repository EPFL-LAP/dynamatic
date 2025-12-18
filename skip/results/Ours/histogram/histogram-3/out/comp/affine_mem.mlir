module {
  func.func @histogram(%arg0: memref<1000xi32> {handshake.arg_name = "feature"}, %arg1: memref<1000xf32> {handshake.arg_name = "weight"}, %arg2: memref<1000xf32> {handshake.arg_name = "hist"}, %arg3: i32 {handshake.arg_name = "n"}) {
    %0 = arith.index_cast %arg3 {handshake.name = "index_cast0"} : i32 to index
    affine.for %arg4 = 0 to %0 {
      %1 = affine.load %arg0[%arg4] {handshake.name = "load0"} : memref<1000xi32>
      %2 = affine.load %arg1[%arg4] {handshake.name = "load1"} : memref<1000xf32>
      %3 = arith.index_cast %1 {handshake.name = "index_cast1"} : i32 to index
      %4 = memref.load %arg2[%3] {handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.name = "load2"} : memref<1000xf32>
      %5 = arith.addf %4, %2 {handshake.name = "addf0"} : f32
      memref.store %5, %arg2[%3] {handshake.deps = #handshake<deps[["load2", 0, true], ["store0", 0, true]]>, handshake.name = "store0"} : memref<1000xf32>
    } {handshake.name = "for0"}
    return {handshake.name = "return0"}
  }
}

