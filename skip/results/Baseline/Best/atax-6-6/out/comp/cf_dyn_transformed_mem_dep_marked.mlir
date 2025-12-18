module {
  func.func @atax(%arg0: memref<400xf32> {handshake.arg_name = "A"}, %arg1: memref<20xf32> {handshake.arg_name = "x"}, %arg2: memref<20xf32> {handshake.arg_name = "y"}, %arg3: memref<20xf32> {handshake.arg_name = "tmp"}) {
    %c0 = arith.constant {handshake.name = "constant4"} 0 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb4
    %c0_0 = arith.constant {handshake.name = "constant5"} 0 : index
    %1 = memref.load %arg3[%0] {handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : memref<20xf32>
    cf.br ^bb2(%c0_0, %1 : index, f32) {handshake.name = "br1"}
  ^bb2(%2: index, %3: f32):  // 2 preds: ^bb1, ^bb2
    %c0_1 = arith.constant {handshake.name = "constant6"} 0 : index
    %c20 = arith.constant {handshake.name = "constant7"} 20 : index
    %c1 = arith.constant {handshake.name = "constant8"} 1 : index
    %c4 = arith.constant {handshake.name = "constant9"} 4 : index
    %c2 = arith.constant {handshake.name = "constant10"} 2 : index
    %4 = arith.shli %0, %c2 {handshake.name = "shli0"} : index
    %5 = arith.shli %0, %c4 {handshake.name = "shli1"} : index
    %6 = arith.addi %4, %5 {handshake.name = "addi5"} : index
    %7 = arith.addi %2, %6 {handshake.name = "addi0"} : index
    %8 = memref.load %arg0[%7] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : memref<400xf32>
    %9 = memref.load %arg1[%2] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : memref<20xf32>
    %10 = arith.mulf %8, %9 {handshake.name = "mulf0"} : f32
    %11 = arith.addf %3, %10 {handshake.name = "addf0"} : f32
    %12 = arith.addi %2, %c1 {handshake.name = "addi2"} : index
    %13 = arith.cmpi ult, %12, %c20 {handshake.name = "cmpi0"} : index
    cf.cond_br %13, ^bb2(%12, %11 : index, f32), ^bb3(%c0_1 : index) {handshake.name = "cond_br0"}
  ^bb3(%14: index):  // 2 preds: ^bb2, ^bb3
    %c20_2 = arith.constant {handshake.name = "constant11"} 20 : index
    %c1_3 = arith.constant {handshake.name = "constant12"} 1 : index
    %c4_4 = arith.constant {handshake.name = "constant13"} 4 : index
    %c2_5 = arith.constant {handshake.name = "constant14"} 2 : index
    %15 = memref.load %arg2[%14] {handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : memref<20xf32>
    %16 = arith.shli %0, %c2_5 {handshake.name = "shli2"} : index
    %17 = arith.shli %0, %c4_4 {handshake.name = "shli3"} : index
    %18 = arith.addi %16, %17 {handshake.name = "addi6"} : index
    %19 = arith.addi %14, %18 {handshake.name = "addi1"} : index
    %20 = memref.load %arg0[%19] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : memref<400xf32>
    %21 = arith.mulf %20, %11 {handshake.name = "mulf1"} : f32
    %22 = arith.addf %15, %21 {handshake.name = "addf1"} : f32
    memref.store %22, %arg2[%14] {handshake.deps = #handshake<deps[["load3", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : memref<20xf32>
    %23 = arith.addi %14, %c1_3 {handshake.name = "addi3"} : index
    %24 = arith.cmpi ult, %23, %c20_2 {handshake.name = "cmpi1"} : index
    cf.cond_br %24, ^bb3(%23 : index), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    %c20_6 = arith.constant {handshake.name = "constant15"} 20 : index
    %c1_7 = arith.constant {handshake.name = "constant16"} 1 : index
    memref.store %11, %arg3[%0] {handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : memref<20xf32>
    %25 = arith.addi %0, %c1_7 {handshake.name = "addi4"} : index
    %26 = arith.cmpi ult, %25, %c20_6 {handshake.name = "cmpi2"} : index
    cf.cond_br %26, ^bb1(%25 : index), ^bb5 {handshake.name = "cond_br2"}
  ^bb5:  // pred: ^bb4
    return {handshake.name = "return0"}
  }
}

