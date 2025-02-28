module {
  func.func @single_loop(%arg0: memref<1000xi32> {handshake.arg_name = "a"}, %arg1: memref<1000xi32> {handshake.arg_name = "b"}, %arg2: memref<1000xi32> {handshake.arg_name = "c"}) {
    %c1000_i32 = arith.constant {handshake.name = "constant0"} 1000 : i32
    %c0_i32 = arith.constant {handshake.name = "constant1"} 0 : i32
    %c1_i32 = arith.constant {handshake.name = "constant2"} 1 : i32
    cf.br ^bb1(%c0_i32 : i32) {handshake.name = "br0"}
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : i32 to index
    %2 = memref.load %arg0[%1] {handshake.name = "load0"} : memref<1000xi32>
    %3 = memref.load %arg1[%1] {handshake.name = "load1"} : memref<1000xi32>
    %4 = arith.muli %2, %3 {handshake.name = "muli0"} : i32
    memref.store %4, %arg2[%1] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "store0"} : memref<1000xi32>
    %5 = arith.cmpi slt, %4, %c1000_i32 {handshake.name = "cmpi0"} : i32
    %6 = arith.addi %0, %c1_i32 {handshake.name = "addi0"} : i32
    cf.cond_br %5, ^bb1(%6 : i32), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

