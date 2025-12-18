module {
  func.func @kernel_3mm(%arg0: memref<10x10xi32> {handshake.arg_name = "A"}, %arg1: memref<10x10xi32> {handshake.arg_name = "B"}, %arg2: memref<10x10xi32> {handshake.arg_name = "C"}, %arg3: memref<10x10xi32> {handshake.arg_name = "D"}, %arg4: memref<10x10xi32> {handshake.arg_name = "E"}, %arg5: memref<10x10xi32> {handshake.arg_name = "F"}, %arg6: memref<10x10xi32> {handshake.arg_name = "G"}) {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    affine.for %arg7 = 0 to 10 {
      affine.for %arg8 = 0 to 10 {
        affine.store %c0_i32, %arg4[%arg7, %arg8] {handshake.deps = #handshake<deps[["load2", 3, true], ["store1", 3, true], ["load6", 1, true]]>, handshake.name = "store0"} : memref<10x10xi32>
        affine.for %arg9 = 0 to 10 {
          %0 = affine.load %arg0[%arg7, %arg9] {handshake.name = "load0"} : memref<10x10xi32>
          %1 = affine.load %arg1[%arg9, %arg8] {handshake.name = "load1"} : memref<10x10xi32>
          %2 = arith.muli %0, %1 {handshake.name = "muli0"} : i32
          %3 = affine.load %arg4[%arg7, %arg8] {handshake.deps = #handshake<deps[["store1", 3, true], ["store1", 4, true]]>, handshake.name = "load2"} : memref<10x10xi32>
          %4 = arith.addi %3, %2 {handshake.name = "addi0"} : i32
          affine.store %4, %arg4[%arg7, %arg8] {handshake.deps = #handshake<deps[["load2", 3, true], ["store1", 3, true], ["load6", 1, true]]>, handshake.name = "store1"} : memref<10x10xi32>
        } {handshake.name = "for0"}
      } {handshake.name = "for1"}
    } {handshake.name = "for2"}
    affine.for %arg7 = 0 to 10 {
      affine.for %arg8 = 0 to 10 {
        affine.store %c0_i32, %arg5[%arg7, %arg8] {handshake.deps = #handshake<deps[["load5", 3, true], ["store3", 3, true], ["load7", 1, true]]>, handshake.name = "store2"} : memref<10x10xi32>
        affine.for %arg9 = 0 to 10 {
          %0 = affine.load %arg2[%arg7, %arg9] {handshake.name = "load3"} : memref<10x10xi32>
          %1 = affine.load %arg3[%arg9, %arg8] {handshake.name = "load4"} : memref<10x10xi32>
          %2 = arith.muli %0, %1 {handshake.name = "muli1"} : i32
          %3 = affine.load %arg5[%arg7, %arg8] {handshake.deps = #handshake<deps[["store3", 3, true], ["store3", 4, true]]>, handshake.name = "load5"} : memref<10x10xi32>
          %4 = arith.addi %3, %2 {handshake.name = "addi1"} : i32
          affine.store %4, %arg5[%arg7, %arg8] {handshake.deps = #handshake<deps[["load5", 3, true], ["store3", 3, true], ["load7", 1, true]]>, handshake.name = "store3"} : memref<10x10xi32>
        } {handshake.name = "for3"}
      } {handshake.name = "for4"}
    } {handshake.name = "for5"}
    affine.for %arg7 = 0 to 10 {
      affine.for %arg8 = 0 to 10 {
        affine.store %c0_i32, %arg6[%arg7, %arg8] {handshake.deps = #handshake<deps[["load8", 3, true], ["store5", 3, true]]>, handshake.name = "store4"} : memref<10x10xi32>
        affine.for %arg9 = 0 to 10 {
          %0 = affine.load %arg4[%arg7, %arg9] {handshake.name = "load6"} : memref<10x10xi32>
          %1 = affine.load %arg5[%arg9, %arg8] {handshake.name = "load7"} : memref<10x10xi32>
          %2 = arith.muli %0, %1 {handshake.name = "muli2"} : i32
          %3 = affine.load %arg6[%arg7, %arg8] {handshake.deps = #handshake<deps[["store5", 3, true], ["store5", 4, true]]>, handshake.name = "load8"} : memref<10x10xi32>
          %4 = arith.addi %3, %2 {handshake.name = "addi2"} : i32
          affine.store %4, %arg6[%arg7, %arg8] {handshake.deps = #handshake<deps[["load8", 3, true], ["store5", 3, true]]>, handshake.name = "store5"} : memref<10x10xi32>
        } {handshake.name = "for6"}
      } {handshake.name = "for7"}
    } {handshake.name = "for8"}
    return {handshake.name = "return0"}
  }
}

