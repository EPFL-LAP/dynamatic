module {
  func.func @kernel_2mm(%arg0: i32 {handshake.arg_name = "alpha"}, %arg1: i32 {handshake.arg_name = "beta"}, %arg2: memref<100xi32> {handshake.arg_name = "tmp"}, %arg3: memref<100xi32> {handshake.arg_name = "A"}, %arg4: memref<100xi32> {handshake.arg_name = "B"}, %arg5: memref<100xi32> {handshake.arg_name = "C"}, %arg6: memref<100xi32> {handshake.arg_name = "D"}) {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %c10 = arith.constant {handshake.name = "constant2"} 10 : index
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    cf.br ^bb2(%c0 : index) {handshake.name = "br1"}
  ^bb2(%1: index):  // 2 preds: ^bb1, ^bb4
    %2 = arith.muli %0, %c10 {handshake.name = "muli4"} : index
    %3 = arith.addi %1, %2 {handshake.name = "addi2"} : index
    memref.store %c0_i32, %arg2[%3] {handshake.deps = #handshake<deps[["load16", 3, true], ["store9", 3, true], ["load18", 1, true]]>, handshake.name = "store8"} : memref<100xi32>
    cf.br ^bb3(%c0 : index) {handshake.name = "br2"}
  ^bb3(%4: index):  // 2 preds: ^bb2, ^bb3
    %5 = arith.muli %0, %c10 {handshake.name = "muli5"} : index
    %6 = arith.addi %4, %5 {handshake.name = "addi3"} : index
    %7 = memref.load %arg3[%6] {handshake.name = "load14"} : memref<100xi32>
    %8 = arith.muli %arg0, %7 {handshake.name = "muli0"} : i32
    %9 = arith.muli %4, %c10 {handshake.name = "muli6"} : index
    %10 = arith.addi %1, %9 {handshake.name = "addi4"} : index
    %11 = memref.load %arg4[%10] {handshake.name = "load15"} : memref<100xi32>
    %12 = arith.muli %8, %11 {handshake.name = "muli1"} : i32
    %13 = arith.muli %0, %c10 {handshake.name = "muli7"} : index
    %14 = arith.addi %1, %13 {handshake.name = "addi5"} : index
    %15 = memref.load %arg2[%14] {handshake.deps = #handshake<deps[["store9", 3, true], ["store9", 4, true]]>, handshake.name = "load16"} : memref<100xi32>
    %16 = arith.addi %15, %12 {handshake.name = "addi0"} : i32
    %17 = arith.muli %0, %c10 {handshake.name = "muli8"} : index
    %18 = arith.addi %1, %17 {handshake.name = "addi6"} : index
    memref.store %16, %arg2[%18] {handshake.deps = #handshake<deps[["load16", 3, true], ["store9", 3, true], ["load18", 1, true]]>, handshake.name = "store9"} : memref<100xi32>
    %19 = arith.addi %4, %c1 {handshake.name = "addi13"} : index
    %20 = arith.cmpi ult, %19, %c10 {handshake.name = "cmpi0"} : index
    cf.cond_br %20, ^bb3(%19 : index), ^bb4 {handshake.name = "cond_br0"}
  ^bb4:  // pred: ^bb3
    %21 = arith.addi %1, %c1 {handshake.name = "addi14"} : index
    %22 = arith.cmpi ult, %21, %c10 {handshake.name = "cmpi1"} : index
    cf.cond_br %22, ^bb2(%21 : index), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb4
    %23 = arith.addi %0, %c1 {handshake.name = "addi15"} : index
    %24 = arith.cmpi ult, %23, %c10 {handshake.name = "cmpi2"} : index
    cf.cond_br %24, ^bb1(%23 : index), ^bb6 {handshake.name = "cond_br2"}
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%c0 : index) {handshake.name = "br3"}
  ^bb7(%25: index):  // 2 preds: ^bb6, ^bb11
    cf.br ^bb8(%c0 : index) {handshake.name = "br4"}
  ^bb8(%26: index):  // 2 preds: ^bb7, ^bb10
    %27 = arith.muli %25, %c10 {handshake.name = "muli9"} : index
    %28 = arith.addi %26, %27 {handshake.name = "addi7"} : index
    %29 = memref.load %arg6[%28] {handshake.deps = #handshake<deps[["store10", 3, true], ["store11", 3, true]]>, handshake.name = "load17"} : memref<100xi32>
    %30 = arith.muli %29, %arg1 {handshake.name = "muli2"} : i32
    %31 = arith.muli %25, %c10 {handshake.name = "muli10"} : index
    %32 = arith.addi %26, %31 {handshake.name = "addi8"} : index
    memref.store %30, %arg6[%32] {handshake.deps = #handshake<deps[["load20", 3, true], ["store11", 3, true]]>, handshake.name = "store10"} : memref<100xi32>
    cf.br ^bb9(%c0 : index) {handshake.name = "br5"}
  ^bb9(%33: index):  // 2 preds: ^bb8, ^bb9
    %34 = arith.muli %25, %c10 {handshake.name = "muli11"} : index
    %35 = arith.addi %33, %34 {handshake.name = "addi9"} : index
    %36 = memref.load %arg2[%35] {handshake.name = "load18"} : memref<100xi32>
    %37 = arith.muli %33, %c10 {handshake.name = "muli12"} : index
    %38 = arith.addi %26, %37 {handshake.name = "addi10"} : index
    %39 = memref.load %arg5[%38] {handshake.name = "load19"} : memref<100xi32>
    %40 = arith.muli %36, %39 {handshake.name = "muli3"} : i32
    %41 = arith.muli %25, %c10 {handshake.name = "muli13"} : index
    %42 = arith.addi %26, %41 {handshake.name = "addi11"} : index
    %43 = memref.load %arg6[%42] {handshake.deps = #handshake<deps[["store11", 3, true], ["store11", 4, true]]>, handshake.name = "load20"} : memref<100xi32>
    %44 = arith.addi %43, %40 {handshake.name = "addi1"} : i32
    %45 = arith.muli %25, %c10 {handshake.name = "muli14"} : index
    %46 = arith.addi %26, %45 {handshake.name = "addi12"} : index
    memref.store %44, %arg6[%46] {handshake.deps = #handshake<deps[["load20", 3, true], ["store11", 3, true]]>, handshake.name = "store11"} : memref<100xi32>
    %47 = arith.addi %33, %c1 {handshake.name = "addi16"} : index
    %48 = arith.cmpi ult, %47, %c10 {handshake.name = "cmpi3"} : index
    cf.cond_br %48, ^bb9(%47 : index), ^bb10 {handshake.name = "cond_br3"}
  ^bb10:  // pred: ^bb9
    %49 = arith.addi %26, %c1 {handshake.name = "addi17"} : index
    %50 = arith.cmpi ult, %49, %c10 {handshake.name = "cmpi4"} : index
    cf.cond_br %50, ^bb8(%49 : index), ^bb11 {handshake.name = "cond_br4"}
  ^bb11:  // pred: ^bb10
    %51 = arith.addi %25, %c1 {handshake.name = "addi18"} : index
    %52 = arith.cmpi ult, %51, %c10 {handshake.name = "cmpi5"} : index
    cf.cond_br %52, ^bb7(%51 : index), ^bb12 {handshake.name = "cond_br5"}
  ^bb12:  // pred: ^bb11
    return {handshake.name = "return0"}
  }
}

