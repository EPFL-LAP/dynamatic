module {
  func.func @jacobi_1d_imper(%arg0: memref<100xi32> {handshake.arg_name = "A"}, %arg1: memref<100xi32> {handshake.arg_name = "B"}) {
    %c3_i32 = arith.constant {handshake.name = "constant0"} 3 : i32
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 1 to 99 {
        %0 = affine.load %arg0[%arg3 - 1] {handshake.deps = #handshake<deps[["store1", 1, true], ["store1", 2, true]]>, handshake.name = "load0"} : memref<100xi32>
        %1 = affine.load %arg0[%arg3] {handshake.deps = #handshake<deps[["store1", 1, true], ["store1", 2, true]]>, handshake.name = "load1"} : memref<100xi32>
        %2 = arith.addi %0, %1 {handshake.name = "addi0"} : i32
        %3 = affine.load %arg0[%arg3 + 1] {handshake.deps = #handshake<deps[["store1", 1, true], ["store1", 2, true]]>, handshake.name = "load2"} : memref<100xi32>
        %4 = arith.addi %2, %3 {handshake.name = "addi1"} : i32
        %5 = arith.muli %4, %c3_i32 {handshake.name = "muli0"} : i32
        affine.store %5, %arg1[%arg3] {handshake.deps = #handshake<deps[["store0", 1, true], ["load3", 1, true], ["load3", 2, true]]>, handshake.name = "store0"} : memref<100xi32>
      } {handshake.name = "for0"}
      affine.for %arg3 = 1 to 99 {
        %0 = affine.load %arg1[%arg3] {handshake.deps = #handshake<deps[["store0", 1, true]]>, handshake.name = "load3"} : memref<100xi32>
        affine.store %0, %arg0[%arg3] {handshake.deps = #handshake<deps[["load0", 1, true], ["load1", 1, true], ["load2", 1, true], ["store1", 1, true]]>, handshake.name = "store1"} : memref<100xi32>
      } {handshake.name = "for1"}
    } {handshake.name = "for2"}
    return {handshake.name = "return0"}
  }
}

