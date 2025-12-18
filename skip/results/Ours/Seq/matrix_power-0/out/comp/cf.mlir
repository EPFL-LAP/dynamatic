module {
  func.func @matrix_power(%arg0: memref<400xi32> {handshake.arg_name = "mat"}, %arg1: memref<20xi32> {handshake.arg_name = "row"}, %arg2: memref<20xi32> {handshake.arg_name = "col"}, %arg3: memref<20xi32> {handshake.arg_name = "a"}) {
    %c0 = arith.constant {handshake.name = "constant4"} 0 : index
    %c-1_i32 = arith.constant {handshake.name = "constant0"} -1 : i32
    %c1 = arith.constant {handshake.name = "constant1"} 1 : index
    %c20 = arith.constant {handshake.name = "constant2"} 20 : index
    cf.br ^bb1(%c1 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb3
    %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : index to i32
    %2 = arith.addi %1, %c-1_i32 {handshake.name = "addi0"} : i32
    %3 = arith.index_cast %2 {handshake.name = "index_cast1"} : i32 to index
    cf.br ^bb2(%c0 : index) {handshake.name = "br1"}
  ^bb2(%4: index):  // 2 preds: ^bb1, ^bb2
    %5 = memref.load %arg1[%4] {handshake.name = "load5"} : memref<20xi32>
    %6 = arith.index_cast %5 {handshake.name = "index_cast2"} : i32 to index
    %7 = memref.load %arg3[%4] {handshake.name = "load6"} : memref<20xi32>
    %8 = memref.load %arg2[%4] {handshake.name = "load7"} : memref<20xi32>
    %9 = arith.index_cast %8 {handshake.name = "index_cast3"} : i32 to index
    %10 = arith.muli %3, %c20 {handshake.name = "muli1"} : index
    %11 = arith.addi %9, %10 {handshake.name = "addi2"} : index
    %12 = memref.load %arg0[%11] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.name = "load8"} : memref<400xi32>
    %13 = arith.muli %7, %12 {handshake.name = "muli0"} : i32
    %14 = arith.muli %0, %c20 {handshake.name = "muli2"} : index
    %15 = arith.addi %6, %14 {handshake.name = "addi3"} : index
    %16 = memref.load %arg0[%15] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.name = "load9"} : memref<400xi32>
    %17 = arith.addi %16, %13 {handshake.name = "addi1"} : i32
    %18 = arith.muli %0, %c20 {handshake.name = "muli3"} : index
    %19 = arith.addi %6, %18 {handshake.name = "addi4"} : index
    memref.store %17, %arg0[%19] {handshake.deps = #handshake<deps[["load8", 0, true], ["load9", 0, true], ["store1", 0, true]]>, handshake.name = "store1"} : memref<400xi32>
    %20 = arith.addi %4, %c1 {handshake.name = "addi5"} : index
    %21 = arith.cmpi ult, %20, %c20 {handshake.name = "cmpi0"} : index
    cf.cond_br %21, ^bb2(%20 : index), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    %22 = arith.addi %0, %c1 {handshake.name = "addi6"} : index
    %23 = arith.cmpi ult, %22, %c20 {handshake.name = "cmpi1"} : index
    cf.cond_br %23, ^bb1(%22 : index), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    return {handshake.name = "return0"}
  }
}

