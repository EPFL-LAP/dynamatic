module {
  func.func @gaussian(%arg0: memref<20xi32> {handshake.arg_name = "c"}, %arg1: memref<400xi32> {handshake.arg_name = "a"}) -> i32 {
    %c0_i32 = arith.constant {handshake.name = "constant2"} 0 : i32
    %c1 = arith.constant {handshake.name = "constant4"} 1 : index
    cf.br ^bb1(%c1, %c0_i32 : index, i32) {handshake.name = "br0"}
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb5
    %c1_0 = arith.constant {handshake.name = "constant7"} 1 : index
    %2 = arith.addi %0, %c1_0 {handshake.name = "addi2"} : index
    cf.br ^bb2(%2, %1 : index, i32) {handshake.name = "br1"}
  ^bb2(%3: index, %4: i32):  // 2 preds: ^bb1, ^bb4
    %c19 = arith.constant {handshake.name = "constant8"} 19 : index
    %c1_1 = arith.constant {handshake.name = "constant9"} 1 : index
    %c1_i32 = arith.constant {handshake.name = "constant10"} 1 : i32
    %5 = arith.cmpi ult, %3, %c19 {handshake.name = "cmpi2"} : index
    cf.cond_br %5, ^bb3(%c1_1, %c1_i32, %4 : index, i32, i32), ^bb5 {handshake.name = "cond_br0"}
  ^bb3(%6: index, %7: i32, %8: i32):  // 2 preds: ^bb2, ^bb3
    %c20 = arith.constant {handshake.name = "constant11"} 20 : index
    %c1_2 = arith.constant {handshake.name = "constant12"} 1 : index
    %c1_i32_3 = arith.constant {handshake.name = "constant13"} 1 : i32
    %c4 = arith.constant {handshake.name = "constant14"} 4 : index
    %c2 = arith.constant {handshake.name = "constant15"} 2 : index
    %9 = arith.index_cast %7 {handshake.name = "index_cast0"} : i32 to index
    %10 = arith.shli %3, %c2 {handshake.name = "shli0"} : index
    %11 = arith.shli %3, %c4 {handshake.name = "shli1"} : index
    %12 = arith.addi %10, %11 {handshake.name = "addi9"} : index
    %13 = arith.addi %9, %12 {handshake.name = "addi3"} : index
    %14 = memref.load %arg1[%13] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load0"} : memref<400xi32>
    %15 = memref.load %arg0[%0] {handshake.name = "load1"} : memref<20xi32>
    %16 = arith.shli %0, %c2 {handshake.name = "shli2"} : index
    %17 = arith.shli %0, %c4 {handshake.name = "shli3"} : index
    %18 = arith.addi %16, %17 {handshake.name = "addi10"} : index
    %19 = arith.addi %9, %18 {handshake.name = "addi4"} : index
    %20 = memref.load %arg1[%19] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load2"} : memref<400xi32>
    %21 = arith.muli %15, %20 {handshake.name = "muli0"} : i32
    %22 = arith.subi %14, %21 {handshake.name = "subi0"} : i32
    %23 = arith.shli %3, %c2 {handshake.name = "shli4"} : index
    %24 = arith.shli %3, %c4 {handshake.name = "shli5"} : index
    %25 = arith.addi %23, %24 {handshake.name = "addi11"} : index
    %26 = arith.addi %9, %25 {handshake.name = "addi5"} : index
    memref.store %22, %arg1[%26] {handshake.deps = #handshake<deps[["load0", 0], ["load2", 0], ["store0", 0]]>, handshake.name = "store0"} : memref<400xi32>
    %27 = arith.addi %8, %7 {handshake.name = "addi0"} : i32
    %28 = arith.addi %7, %c1_i32_3 {handshake.name = "addi1"} : i32
    %29 = arith.addi %6, %c1_2 {handshake.name = "addi6"} : index
    %30 = arith.cmpi ult, %29, %c20 {handshake.name = "cmpi0"} : index
    cf.cond_br %30, ^bb3(%29, %28, %27 : index, i32, i32), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    %c1_4 = arith.constant {handshake.name = "constant16"} 1 : index
    %31 = arith.addi %3, %c1_4 {handshake.name = "addi8"} : index
    cf.br ^bb2(%31, %27 : index, i32) {handshake.name = "br3"}
  ^bb5:  // pred: ^bb2
    %c19_5 = arith.constant {handshake.name = "constant17"} 19 : index
    %c1_6 = arith.constant {handshake.name = "constant18"} 1 : index
    %32 = arith.addi %0, %c1_6 {handshake.name = "addi7"} : index
    %33 = arith.cmpi ult, %32, %c19_5 {handshake.name = "cmpi1"} : index
    cf.cond_br %33, ^bb1(%32, %4 : index, i32), ^bb6 {handshake.name = "cond_br2"}
  ^bb6:  // pred: ^bb5
    return {handshake.name = "return0"} %4 : i32
  }
}

