module {
  func.func @get_tanh(%arg0: memref<1000xf32> {handshake.arg_name = "A"}, %arg1: memref<1000xi32> {handshake.arg_name = "addr"}) {
    %c0 = arith.constant {handshake.name = "constant2"} 0 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb3
    %cst = arith.constant {handshake.name = "constant6"} 1.000000e+00 : f32
    %1 = memref.load %arg1[%0] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : memref<1000xi32>
    %2 = arith.index_cast %1 {handshake.name = "index_cast0"} : i32 to index
    %3 = memref.load %arg0[%2] {handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load1"} : memref<1000xf32>
    %4 = arith.cmpf oge, %3, %cst {handshake.name = "cmpf0"} : f32
    cf.cond_br %4, ^bb3(%cst : f32), ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %cst_0 = arith.constant {handshake.name = "constant7"} 3.70476198 : f32
    %cst_1 = arith.constant {handshake.name = "constant8"} 19.5238094 : f32
    %5 = arith.mulf %3, %3 {handshake.name = "mulf0"} : f32
    %6 = arith.addf %5, %cst_1 {handshake.name = "addf0"} : f32
    %7 = arith.mulf %6, %3 {handshake.name = "mulf1"} : f32
    %8 = arith.mulf %7, %3 {handshake.name = "mulf2"} : f32
    %9 = arith.addf %8, %cst_0 {handshake.name = "addf1"} : f32
    %10 = arith.mulf %9, %3 {handshake.name = "mulf3"} : f32
    cf.br ^bb3(%10 : f32) {handshake.name = "br2"}
  ^bb3(%11: f32):  // 2 preds: ^bb1, ^bb2
    %c1000 = arith.constant {handshake.name = "constant9"} 1000 : index
    %c1 = arith.constant {handshake.name = "constant10"} 1 : index
    memref.store %11, %arg0[%2] {handshake.deps = #handshake<deps[["load1", 0, true], ["store0", 0, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : memref<1000xf32>
    %12 = arith.addi %0, %c1 {handshake.name = "addi0"} : index
    %13 = arith.cmpi ult, %12, %c1000 {handshake.name = "cmpi0"} : index
    cf.cond_br %13, ^bb1(%12 : index), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    return {handshake.name = "return0"}
  }
}

