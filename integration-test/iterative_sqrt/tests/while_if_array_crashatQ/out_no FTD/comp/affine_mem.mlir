module {
  func.func @iterative_sqrt(%arg0: memref<10xi32> {handshake.arg_name = "A"}) -> i32 {
    %c-1_i32 = arith.constant {handshake.name = "constant0"} -1 : i32
    %c1_i32 = arith.constant {handshake.name = "constant1"} 1 : i32
    %c10_i32 = arith.constant {handshake.name = "constant2"} 10 : i32
    %0 = scf.while : () -> i32 {
      %2 = affine.load %arg0[0] {handshake.name = "load0"} : memref<10xi32>
      %3 = arith.cmpi sgt, %2, %c10_i32 {handshake.name = "cmpi0"} : i32
      scf.condition(%3) {handshake.name = "condition0"} %2 : i32
    } do {
    ^bb0(%arg1: i32):
      %2 = affine.load %arg0[1] {handshake.name = "load1"} : memref<10xi32>
      %3 = arith.cmpi slt, %2, %c10_i32 {handshake.name = "cmpi1"} : i32
      scf.if %3 {
        %4 = arith.addi %arg1, %c-1_i32 {handshake.name = "addi0"} : i32
        affine.store %4, %arg0[0] {handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.name = "store0"} : memref<10xi32>
      } else {
        %4 = arith.shrsi %arg1, %c1_i32 {handshake.name = "shrsi0"} : i32
        affine.store %4, %arg0[0] {handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.name = "store1"} : memref<10xi32>
      } {handshake.name = "if0"}
      scf.yield {handshake.name = "yield2"}
    } attributes {handshake.name = "while0"}
    %1 = affine.load %arg0[0] {handshake.name = "load2"} : memref<10xi32>
    return {handshake.name = "return0"} %1 : i32
  }
}

