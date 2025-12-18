#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
module {
  func.func @triangular(%arg0: memref<10xi32> {handshake.arg_name = "x"}, %arg1: i32 {handshake.arg_name = "n"}, %arg2: memref<10x10xi32> {handshake.arg_name = "a"}) {
    %0 = arith.index_cast %arg1 {handshake.name = "index_cast0"} : i32 to index
    affine.for %arg3 = 0 to %0 {
      affine.for %arg4 = 0 to #map(%arg3)[%0] {
        %1 = affine.load %arg2[-%arg3 - %arg4 + symbol(%0) - 2, -%arg3 + symbol(%0) - 1] {handshake.name = "load0"} : memref<10x10xi32>
        %2 = affine.load %arg0[-%arg3 + symbol(%0) - 1] {handshake.name = "load1"} : memref<10xi32>
        %3 = arith.muli %1, %2 {handshake.name = "muli0"} : i32
        %4 = affine.load %arg2[-%arg3 - %arg4 + symbol(%0) - 2, symbol(%0)] {handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.name = "load2"} : memref<10x10xi32>
        %5 = arith.subi %4, %3 {handshake.name = "subi0"} : i32
        affine.store %5, %arg2[-%arg3 - %arg4 + symbol(%0) - 2, symbol(%0)] {handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.name = "store0"} : memref<10x10xi32>
      } {handshake.name = "for0"}
    } {handshake.name = "for1"}
    return {handshake.name = "return0"}
  }
}

