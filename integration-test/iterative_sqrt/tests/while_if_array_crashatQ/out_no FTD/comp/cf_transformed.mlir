module {
  func.func @iterative_sqrt(%arg0: memref<10xi32> {handshake.arg_name = "A"}) -> i32 {
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1 {handshake.name = "br0"}
  ^bb1:  // 3 preds: ^bb0, ^bb3, ^bb4
    %0 = memref.load %arg0[%c0] {handshake.name = "load3"} : memref<10xi32>
    %1 = arith.cmpi sgt, %0, %c10_i32 {handshake.name = "cmpi0"} : i32
    cf.cond_br %1, ^bb2(%0 : i32), ^bb5 {handshake.name = "cond_br0"}
  ^bb2(%2: i32):  // pred: ^bb1
    %3 = memref.load %arg0[%c1] {handshake.name = "load4"} : memref<10xi32>
    %4 = arith.cmpi slt, %3, %c10_i32 {handshake.name = "cmpi1"} : i32
    cf.cond_br %4, ^bb3, ^bb4 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %5 = arith.addi %2, %c-1_i32 {handshake.name = "addi0"} : i32
    memref.store %5, %arg0[%c0] {handshake.deps = #handshake<deps[<"load5" (1)>]>, handshake.name = "store2"} : memref<10xi32>
    cf.br ^bb1
  ^bb4:  // pred: ^bb2
    %6 = arith.shrsi %2, %c1_i32 {handshake.name = "shrsi0"} : i32
    memref.store %6, %arg0[%c0] {handshake.deps = #handshake<deps[<"load5" (1)>]>, handshake.name = "store3"} : memref<10xi32>
    cf.br ^bb1 {handshake.name = "br3"}
  ^bb5:  // pred: ^bb1
    %7 = memref.load %arg0[%c0] {handshake.name = "load5"} : memref<10xi32>
    return {handshake.name = "return0"} %7 : i32
  }
}

