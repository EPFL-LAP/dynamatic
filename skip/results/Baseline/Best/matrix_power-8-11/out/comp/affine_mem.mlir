module {
  func.func @matrix_power(%arg0: memref<20x20xi32> {handshake.arg_name = "mat"}, %arg1: memref<20xi32> {handshake.arg_name = "row"}, %arg2: memref<20xi32> {handshake.arg_name = "col"}, %arg3: memref<20xi32> {handshake.arg_name = "a"}) {
    %c-1_i32 = arith.constant {handshake.name = "constant0"} -1 : i32
    affine.for %arg4 = 1 to 20 {
      %0 = arith.index_cast %arg4 {handshake.name = "index_cast0"} : index to i32
      %1 = arith.addi %0, %c-1_i32 {handshake.name = "addi0"} : i32
      %2 = arith.index_cast %1 {handshake.name = "index_cast1"} : i32 to index
      affine.for %arg5 = 0 to 20 {
        %3 = affine.load %arg1[%arg5] {handshake.name = "load0"} : memref<20xi32>
        %4 = arith.index_cast %3 {handshake.name = "index_cast2"} : i32 to index
        %5 = affine.load %arg3[%arg5] {handshake.name = "load1"} : memref<20xi32>
        %6 = affine.load %arg2[%arg5] {handshake.name = "load2"} : memref<20xi32>
        %7 = arith.index_cast %6 {handshake.name = "index_cast3"} : i32 to index
        %8 = memref.load %arg0[%2, %7] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load3"} : memref<20x20xi32>
        %9 = arith.muli %5, %8 {handshake.name = "muli0"} : i32
        %10 = memref.load %arg0[%arg4, %4] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load4"} : memref<20x20xi32>
        %11 = arith.addi %10, %9 {handshake.name = "addi1"} : i32
        memref.store %11, %arg0[%arg4, %4] {handshake.deps = #handshake<deps[["load3", 0], ["load4", 0], ["store0", 0]]>, handshake.name = "store0"} : memref<20x20xi32>
      } {handshake.name = "for0"}
    } {handshake.name = "for1"}
    return {handshake.name = "return0"}
  }
}

