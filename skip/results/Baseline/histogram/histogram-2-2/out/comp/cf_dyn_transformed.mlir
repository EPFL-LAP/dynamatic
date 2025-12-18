module {
  func.func @histogram(%arg0: memref<1000xi32> {handshake.arg_name = "feature"}, %arg1: memref<1000xf32> {handshake.arg_name = "weight"}, %arg2: memref<1000xf32> {handshake.arg_name = "hist"}, %arg3: i32 {handshake.arg_name = "n"}) {
    %c0 = arith.constant {handshake.name = "constant0"} 0 : index
    %0 = arith.index_cast %arg3 {handshake.name = "index_cast0"} : i32 to index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %1, %0 {handshake.name = "cmpi0"} : index
    cf.cond_br %2, ^bb2, ^bb3 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %c1 = arith.constant {handshake.name = "constant2"} 1 : index
    %3 = memref.load %arg0[%1] {handshake.name = "load0"} : memref<1000xi32>
    %4 = memref.load %arg1[%1] {handshake.name = "load1"} : memref<1000xf32>
    %5 = arith.index_cast %3 {handshake.name = "index_cast1"} : i32 to index
    %6 = memref.load %arg2[%5] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load2"} : memref<1000xf32>
    %7 = arith.addf %6, %4 {handshake.name = "addf0"} : f32
    memref.store %7, %arg2[%5] {handshake.deps = #handshake<deps[["load2", 0], ["store0", 0]]>, handshake.name = "store0"} : memref<1000xf32>
    %8 = arith.addi %1, %c1 {handshake.name = "addi0"} : index
    cf.br ^bb1(%8 : index) {handshake.name = "br1"}
  ^bb3:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

