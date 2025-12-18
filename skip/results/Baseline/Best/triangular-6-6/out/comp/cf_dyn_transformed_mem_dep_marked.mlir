module {
  func.func @triangular(%arg0: memref<10xi32> {handshake.arg_name = "x"}, %arg1: i32 {handshake.arg_name = "n"}, %arg2: memref<100xi32> {handshake.arg_name = "a"}) {
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %0 = arith.index_cast %arg1 {handshake.name = "index_cast0"} : i32 to index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb5
    %2 = arith.cmpi slt, %1, %0 {handshake.name = "cmpi0"} : index
    cf.cond_br %2, ^bb2, ^bb6 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %c-1 = arith.constant {handshake.name = "constant5"} -1 : index
    %c0_0 = arith.constant {handshake.name = "constant6"} 0 : index
    %3 = arith.subi %0, %1 {handshake.name = "subi1"} : index
    %4 = arith.addi %3, %c-1 {handshake.name = "addi1"} : index
    cf.br ^bb3(%c0_0 : index) {handshake.name = "br1"}
  ^bb3(%5: index):  // 2 preds: ^bb2, ^bb4
    %6 = arith.cmpi slt, %5, %4 {handshake.name = "cmpi1"} : index
    cf.cond_br %6, ^bb4, ^bb5 {handshake.name = "cond_br1"}
  ^bb4:  // pred: ^bb3
    %c-2 = arith.constant {handshake.name = "constant7"} -2 : index
    %c-1_1 = arith.constant {handshake.name = "constant8"} -1 : index
    %c1 = arith.constant {handshake.name = "constant9"} 1 : index
    %c3 = arith.constant {handshake.name = "constant10"} 3 : index
    %7 = arith.addi %1, %5 {handshake.name = "addi0"} : index
    %8 = arith.xori %7, %c-1_1 {handshake.name = "xori0"} : index
    %9 = arith.addi %8, %c1 {handshake.name = "addi2"} : index
    %10 = arith.addi %9, %0 {handshake.name = "addi3"} : index
    %11 = arith.addi %10, %c-2 {handshake.name = "addi4"} : index
    %12 = arith.addi %3, %c-1_1 {handshake.name = "addi6"} : index
    %13 = arith.shli %11, %c1 {handshake.name = "shli0"} : index
    %14 = arith.shli %11, %c3 {handshake.name = "shli1"} : index
    %15 = arith.addi %13, %14 {handshake.name = "addi5"} : index
    %16 = arith.addi %12, %15 {handshake.name = "addi15"} : index
    %17 = memref.load %arg2[%16] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : memref<100xi32>
    %18 = arith.addi %3, %c-1_1 {handshake.name = "addi8"} : index
    %19 = memref.load %arg0[%18] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : memref<10xi32>
    %20 = arith.muli %17, %19 {handshake.name = "muli0"} : i32
    %21 = arith.addi %1, %5 {handshake.name = "addi7"} : index
    %22 = arith.xori %21, %c-1_1 {handshake.name = "xori1"} : index
    %23 = arith.addi %22, %c1 {handshake.name = "addi9"} : index
    %24 = arith.addi %23, %0 {handshake.name = "addi10"} : index
    %25 = arith.addi %24, %c-2 {handshake.name = "addi11"} : index
    %26 = arith.shli %25, %c1 {handshake.name = "shli2"} : index
    %27 = arith.shli %25, %c3 {handshake.name = "shli3"} : index
    %28 = arith.addi %26, %27 {handshake.name = "addi12"} : index
    %29 = arith.addi %0, %28 {handshake.name = "addi16"} : index
    %30 = memref.load %arg2[%29] {handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : memref<100xi32>
    %31 = arith.subi %30, %20 {handshake.name = "subi0"} : i32
    %32 = arith.addi %1, %5 {handshake.name = "addi20"} : index
    %33 = arith.xori %32, %c-1_1 {handshake.name = "xori2"} : index
    %34 = arith.addi %33, %c1 {handshake.name = "addi21"} : index
    %35 = arith.addi %34, %0 {handshake.name = "addi13"} : index
    %36 = arith.addi %35, %c-2 {handshake.name = "addi14"} : index
    %37 = arith.shli %36, %c1 {handshake.name = "shli4"} : index
    %38 = arith.shli %36, %c3 {handshake.name = "shli5"} : index
    %39 = arith.addi %37, %38 {handshake.name = "addi22"} : index
    %40 = arith.addi %0, %39 {handshake.name = "addi17"} : index
    memref.store %31, %arg2[%40] {handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : memref<100xi32>
    %41 = arith.addi %5, %c1 {handshake.name = "addi18"} : index
    cf.br ^bb3(%41 : index) {handshake.name = "br2"}
  ^bb5:  // pred: ^bb3
    %c1_2 = arith.constant {handshake.name = "constant11"} 1 : index
    %42 = arith.addi %1, %c1_2 {handshake.name = "addi19"} : index
    cf.br ^bb1(%42 : index) {handshake.name = "br3"}
  ^bb6:  // pred: ^bb1
    return {handshake.name = "return0"}
  }
}

