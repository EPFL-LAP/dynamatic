#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @gaussian(%arg0: memref<20xi32> {handshake.arg_name = "c"}, %arg1: memref<20x20xi32> {handshake.arg_name = "a"}) -> i32 {
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant1"} 0 : i32
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    %1 = affine.for %arg2 = 1 to 19 iter_args(%arg3 = %c0_i32) -> (i32) {
      %2 = affine.for %arg4 = #map(%arg2) to 19 iter_args(%arg5 = %arg3) -> (i32) {
        %3:6 = affine.for %arg6 = 1 to 20 iter_args(%arg7 = %c1_i32, %arg8 = %arg5, %arg9 = %arg5, %arg10 = %0, %arg11 = %c1_i32, %arg12 = %arg5) -> (i32, i32, i32, i32, i32, i32) {
          %4 = arith.index_cast %arg7 {handshake.name = "index_cast0"} : i32 to index
          %5 = memref.load %arg1[%arg4, %4] {handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.name = "load0"} : memref<20x20xi32>
          %6 = affine.load %arg0[%arg2] {handshake.name = "load1"} : memref<20xi32>
          %7 = memref.load %arg1[%arg2, %4] {handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.name = "load2"} : memref<20x20xi32>
          %8 = arith.muli %6, %7 {handshake.name = "muli0"} : i32
          %9 = arith.subi %5, %8 {handshake.name = "subi0"} : i32
          memref.store %9, %arg1[%arg4, %4] {handshake.deps = #handshake<deps[["load0", 0, true], ["load2", 0, true], ["store0", 0, true]]>, handshake.name = "store0"} : memref<20x20xi32>
          %10 = arith.addi %arg8, %arg7 {handshake.name = "addi0"} : i32
          %11 = arith.addi %arg7, %c1_i32 {handshake.name = "addi1"} : i32
          affine.yield {handshake.name = "yield0"} %11, %10, %10, %0, %11, %10 : i32, i32, i32, i32, i32, i32
        } {handshake.name = "for0"}
        affine.yield {handshake.name = "yield1"} %3#2 : i32
      } {handshake.name = "for1"}
      affine.yield {handshake.name = "yield2"} %2 : i32
    } {handshake.name = "for2"}
    return {handshake.name = "return0"} %1 : i32
  }
}

