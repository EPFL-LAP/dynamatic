module {
  func.func @iterative_sqrt(%arg0: memref<10xi32> {handshake.arg_name = "A"}) -> i32 {
    %c1 = arith.constant {handshake.name = "constant4"} 1 : index
    %c0 = arith.constant {handshake.name = "constant3"} 0 : index
    %c-1_i32 = arith.constant {handshake.name = "constant0"} -1 : i32
    %c1_i32 = arith.constant {handshake.name = "constant1"} 1 : i32
    %c10_i32 = arith.constant {handshake.name = "constant2"} 10 : i32
    %0 = scf.while : () -> i32 {
      %2 = memref.load %arg0[%c0] {handshake.name = "load3"} : memref<10xi32>
      %3 = arith.cmpi sgt, %2, %c10_i32 {handshake.name = "cmpi0"} : i32
      scf.condition(%3) {handshake.name = "condition0"} %2 : i32
    } do {
    ^bb0(%arg1: i32):
      %2 = memref.load %arg0[%c1] {handshake.name = "load4"} : memref<10xi32>
      %3 = arith.cmpi slt, %2, %c10_i32 {handshake.name = "cmpi1"} : i32
      scf.if %3 {
        %4 = arith.addi %arg1, %c-1_i32 {handshake.name = "addi0"} : i32
        memref.store %4, %arg0[%c0] {handshake.deps = #handshake<deps[<"load5" (1)>]>, handshake.name = "store2"} : memref<10xi32>
      } else {
        %4 = arith.shrsi %arg1, %c1_i32 {handshake.name = "shrsi0"} : i32
        memref.store %4, %arg0[%c0] {handshake.deps = #handshake<deps[<"load5" (1)>]>, handshake.name = "store3"} : memref<10xi32>
      } {handshake.name = "if0"}
      scf.yield {handshake.name = "yield2"}
    } attributes {handshake.name = "while0"}
    %1 = memref.load %arg0[%c0] {handshake.name = "load5"} : memref<10xi32>
    return {handshake.name = "return0"} %1 : i32
  }
}

