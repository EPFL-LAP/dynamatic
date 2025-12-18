module {
  func.func @bicg(%arg0: memref<30x30xi32> {handshake.arg_name = "a"}, %arg1: memref<30xi32> {handshake.arg_name = "s"}, %arg2: memref<30xi32> {handshake.arg_name = "q"}, %arg3: memref<30xi32> {handshake.arg_name = "p"}, %arg4: memref<30xi32> {handshake.arg_name = "r"}) -> i32 {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %0 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %c0_i32) -> (i32) {
      %1 = affine.load %arg2[%arg5] {handshake.deps = #handshake<deps[["store1", 2, true]]>, handshake.name = "load0"} : memref<30xi32>
      %2 = affine.for %arg7 = 0 to 30 iter_args(%arg8 = %1) -> (i32) {
        %3 = affine.load %arg0[%arg5, %arg7] {handshake.name = "load1"} : memref<30x30xi32>
        %4 = affine.load %arg1[%arg7] {handshake.deps = #handshake<deps[["store0", 1, true], ["store0", 3, true]]>, handshake.name = "load2"} : memref<30xi32>
        %5 = affine.load %arg4[%arg5] {handshake.name = "load3"} : memref<30xi32>
        %6 = arith.muli %5, %3 {handshake.name = "muli0"} : i32
        %7 = arith.addi %4, %6 {handshake.name = "addi0"} : i32
        affine.store %7, %arg1[%arg7] {handshake.deps = #handshake<deps[["load2", 1, true], ["store0", 1, true]]>, handshake.name = "store0"} : memref<30xi32>
        %8 = affine.load %arg3[%arg7] {handshake.name = "load4"} : memref<30xi32>
        %9 = arith.muli %3, %8 {handshake.name = "muli1"} : i32
        %10 = arith.addi %arg8, %9 {handshake.name = "addi1"} : i32
        affine.yield {handshake.name = "yield0"} %10 : i32
      } {handshake.name = "for0"}
      affine.store %2, %arg2[%arg5] {handshake.name = "store1"} : memref<30xi32>
      affine.yield {handshake.name = "yield1"} %2 : i32
    } {handshake.name = "for1"}
    return {handshake.name = "return0"} %0 : i32
  }
}

