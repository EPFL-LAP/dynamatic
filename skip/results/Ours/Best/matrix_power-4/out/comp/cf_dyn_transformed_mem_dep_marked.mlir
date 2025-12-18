module {
  func.func @matrix_power(%arg0: memref<400xi32> {handshake.arg_name = "mat"}, %arg1: memref<20xi32> {handshake.arg_name = "row"}, %arg2: memref<20xi32> {handshake.arg_name = "col"}, %arg3: memref<20xi32> {handshake.arg_name = "a"}) {
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    cf.br ^bb1(%c1 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb3
    %c0 = arith.constant {handshake.name = "constant6"} 0 : index
    %c-1_i32 = arith.constant {handshake.name = "constant7"} -1 : i32
    %1 = arith.index_cast %0 {handshake.name = "index_cast0"} : index to i32
    %2 = arith.addi %1, %c-1_i32 {handshake.name = "addi0"} : i32
    %3 = arith.index_cast %2 {handshake.name = "index_cast1"} : i32 to index
    cf.br ^bb2(%c0 : index) {handshake.name = "br1"}
  ^bb2(%4: index):  // 2 preds: ^bb1, ^bb2
    %c1_0 = arith.constant {handshake.name = "constant8"} 1 : index
    %c20 = arith.constant {handshake.name = "constant9"} 20 : index
    %c4 = arith.constant {handshake.name = "constant10"} 4 : index
    %c2 = arith.constant {handshake.name = "constant11"} 2 : index
    %5 = memref.load %arg1[%4] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : memref<20xi32>
    %6 = arith.index_cast %5 {handshake.name = "index_cast2"} : i32 to index
    %7 = memref.load %arg3[%4] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load6"} : memref<20xi32>
    %8 = memref.load %arg2[%4] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load7"} : memref<20xi32>
    %9 = arith.index_cast %8 {handshake.name = "index_cast3"} : i32 to index
    %10 = arith.shli %3, %c2 {handshake.name = "shli0"} : index
    %11 = arith.shli %3, %c4 {handshake.name = "shli1"} : index
    %12 = arith.addi %10, %11 {handshake.name = "addi7"} : index
    %13 = arith.addi %9, %12 {handshake.name = "addi2"} : index
    %14 = memref.load %arg0[%13] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load8"} : memref<400xi32>
    %15 = arith.muli %7, %14 {handshake.name = "muli0"} : i32
    %16 = arith.shli %0, %c2 {handshake.name = "shli2"} : index
    %17 = arith.shli %0, %c4 {handshake.name = "shli3"} : index
    %18 = arith.addi %16, %17 {handshake.name = "addi8"} : index
    %19 = arith.addi %6, %18 {handshake.name = "addi3"} : index
    %20 = memref.load %arg0[%19] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load9"} : memref<400xi32>
    %21 = arith.addi %20, %15 {handshake.name = "addi1"} : i32
    %22 = arith.shli %0, %c2 {handshake.name = "shli4"} : index
    %23 = arith.shli %0, %c4 {handshake.name = "shli5"} : index
    %24 = arith.addi %22, %23 {handshake.name = "addi9"} : index
    %25 = arith.addi %6, %24 {handshake.name = "addi4"} : index
    memref.store %21, %arg0[%25] {handshake.deps = #handshake<deps[["load8", 0, true], ["load9", 0, true], ["store1", 0, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store1"} : memref<400xi32>
    %26 = arith.addi %4, %c1_0 {handshake.name = "addi5"} : index
    %27 = arith.cmpi ult, %26, %c20 {handshake.name = "cmpi0"} : index
    cf.cond_br %27, ^bb2(%26 : index), ^bb3 {handshake.name = "cond_br0"}
  ^bb3:  // pred: ^bb2
    %c1_1 = arith.constant {handshake.name = "constant12"} 1 : index
    %c20_2 = arith.constant {handshake.name = "constant13"} 20 : index
    %28 = arith.addi %0, %c1_1 {handshake.name = "addi6"} : index
    %29 = arith.cmpi ult, %28, %c20_2 {handshake.name = "cmpi1"} : index
    cf.cond_br %29, ^bb1(%28 : index), ^bb4 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    return {handshake.name = "return0"}
  }
}

