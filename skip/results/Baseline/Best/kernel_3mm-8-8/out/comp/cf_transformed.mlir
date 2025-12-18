module {
  func.func @kernel_3mm(%arg0: memref<100xi32> {handshake.arg_name = "A"}, %arg1: memref<100xi32> {handshake.arg_name = "B"}, %arg2: memref<100xi32> {handshake.arg_name = "C"}, %arg3: memref<100xi32> {handshake.arg_name = "D"}, %arg4: memref<100xi32> {handshake.arg_name = "E"}, %arg5: memref<100xi32> {handshake.arg_name = "F"}, %arg6: memref<100xi32> {handshake.arg_name = "G"}) {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    cf.br ^bb2(%c0 : index) {handshake.name = "br1"}
  ^bb2(%1: index):  // 2 preds: ^bb1, ^bb4
    %2 = arith.muli %0, %c10 {handshake.name = "muli3"} : index
    %3 = arith.addi %1, %2 {handshake.name = "addi3"} : index
    memref.store %c0_i32, %arg4[%3] {handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load6", 1]]>, handshake.name = "store0"} : memref<100xi32>
    cf.br ^bb3(%c0 : index) {handshake.name = "br2"}
  ^bb3(%4: index):  // 2 preds: ^bb2, ^bb3
    %5 = arith.muli %0, %c10 {handshake.name = "muli4"} : index
    %6 = arith.addi %4, %5 {handshake.name = "addi4"} : index
    %7 = memref.load %arg0[%6] {handshake.name = "load0"} : memref<100xi32>
    %8 = arith.muli %4, %c10 {handshake.name = "muli5"} : index
    %9 = arith.addi %1, %8 {handshake.name = "addi5"} : index
    %10 = memref.load %arg1[%9] {handshake.name = "load1"} : memref<100xi32>
    %11 = arith.muli %7, %10 {handshake.name = "muli0"} : i32
    %12 = arith.muli %0, %c10 {handshake.name = "muli6"} : index
    %13 = arith.addi %1, %12 {handshake.name = "addi6"} : index
    %14 = memref.load %arg4[%13] {handshake.deps = #handshake<deps[["store1", 3], ["store1", 4]]>, handshake.name = "load2"} : memref<100xi32>
    %15 = arith.addi %14, %11 {handshake.name = "addi0"} : i32
    %16 = arith.muli %0, %c10 {handshake.name = "muli7"} : index
    %17 = arith.addi %1, %16 {handshake.name = "addi7"} : index
    memref.store %15, %arg4[%17] {handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load6", 1]]>, handshake.name = "store1"} : memref<100xi32>
    %18 = arith.addi %4, %c1 {handshake.name = "addi18"} : index
    %19 = arith.cmpi ult, %18, %c10 {handshake.name = "cmpi0"} : index
    cf.cond_br %19, ^bb3(%18 : index), ^bb4 {handshake.name = "cond_br0"}
  ^bb4:  // pred: ^bb3
    %20 = arith.addi %1, %c1 {handshake.name = "addi19"} : index
    %21 = arith.cmpi ult, %20, %c10 {handshake.name = "cmpi1"} : index
    cf.cond_br %21, ^bb2(%20 : index), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb4
    %22 = arith.addi %0, %c1 {handshake.name = "addi20"} : index
    %23 = arith.cmpi ult, %22, %c10 {handshake.name = "cmpi2"} : index
    cf.cond_br %23, ^bb1(%22 : index), ^bb6(%c0 : index)
  ^bb6(%24: index):  // 2 preds: ^bb5, ^bb10
    cf.br ^bb7(%c0 : index) {handshake.name = "br4"}
  ^bb7(%25: index):  // 2 preds: ^bb6, ^bb9
    %26 = arith.muli %24, %c10 {handshake.name = "muli8"} : index
    %27 = arith.addi %25, %26 {handshake.name = "addi8"} : index
    memref.store %c0_i32, %arg5[%27] {handshake.deps = #handshake<deps[["load5", 3], ["store3", 3], ["load7", 1]]>, handshake.name = "store2"} : memref<100xi32>
    cf.br ^bb8(%c0 : index) {handshake.name = "br5"}
  ^bb8(%28: index):  // 2 preds: ^bb7, ^bb8
    %29 = arith.muli %24, %c10 {handshake.name = "muli9"} : index
    %30 = arith.addi %28, %29 {handshake.name = "addi9"} : index
    %31 = memref.load %arg2[%30] {handshake.name = "load3"} : memref<100xi32>
    %32 = arith.muli %28, %c10 {handshake.name = "muli10"} : index
    %33 = arith.addi %25, %32 {handshake.name = "addi10"} : index
    %34 = memref.load %arg3[%33] {handshake.name = "load4"} : memref<100xi32>
    %35 = arith.muli %31, %34 {handshake.name = "muli1"} : i32
    %36 = arith.muli %24, %c10 {handshake.name = "muli11"} : index
    %37 = arith.addi %25, %36 {handshake.name = "addi11"} : index
    %38 = memref.load %arg5[%37] {handshake.deps = #handshake<deps[["store3", 3], ["store3", 4]]>, handshake.name = "load5"} : memref<100xi32>
    %39 = arith.addi %38, %35 {handshake.name = "addi1"} : i32
    %40 = arith.muli %24, %c10 {handshake.name = "muli12"} : index
    %41 = arith.addi %25, %40 {handshake.name = "addi12"} : index
    memref.store %39, %arg5[%41] {handshake.deps = #handshake<deps[["load5", 3], ["store3", 3], ["load7", 1]]>, handshake.name = "store3"} : memref<100xi32>
    %42 = arith.addi %28, %c1 {handshake.name = "addi21"} : index
    %43 = arith.cmpi ult, %42, %c10 {handshake.name = "cmpi3"} : index
    cf.cond_br %43, ^bb8(%42 : index), ^bb9 {handshake.name = "cond_br3"}
  ^bb9:  // pred: ^bb8
    %44 = arith.addi %25, %c1 {handshake.name = "addi22"} : index
    %45 = arith.cmpi ult, %44, %c10 {handshake.name = "cmpi4"} : index
    cf.cond_br %45, ^bb7(%44 : index), ^bb10 {handshake.name = "cond_br4"}
  ^bb10:  // pred: ^bb9
    %46 = arith.addi %24, %c1 {handshake.name = "addi23"} : index
    %47 = arith.cmpi ult, %46, %c10 {handshake.name = "cmpi5"} : index
    cf.cond_br %47, ^bb6(%46 : index), ^bb11(%c0 : index)
  ^bb11(%48: index):  // 2 preds: ^bb10, ^bb15
    cf.br ^bb12(%c0 : index) {handshake.name = "br7"}
  ^bb12(%49: index):  // 2 preds: ^bb11, ^bb14
    %50 = arith.muli %48, %c10 {handshake.name = "muli13"} : index
    %51 = arith.addi %49, %50 {handshake.name = "addi13"} : index
    memref.store %c0_i32, %arg6[%51] {handshake.deps = #handshake<deps[["load8", 3], ["store5", 3]]>, handshake.name = "store4"} : memref<100xi32>
    cf.br ^bb13(%c0 : index) {handshake.name = "br8"}
  ^bb13(%52: index):  // 2 preds: ^bb12, ^bb13
    %53 = arith.muli %48, %c10 {handshake.name = "muli14"} : index
    %54 = arith.addi %52, %53 {handshake.name = "addi14"} : index
    %55 = memref.load %arg4[%54] {handshake.name = "load6"} : memref<100xi32>
    %56 = arith.muli %52, %c10 {handshake.name = "muli15"} : index
    %57 = arith.addi %49, %56 {handshake.name = "addi15"} : index
    %58 = memref.load %arg5[%57] {handshake.name = "load7"} : memref<100xi32>
    %59 = arith.muli %55, %58 {handshake.name = "muli2"} : i32
    %60 = arith.muli %48, %c10 {handshake.name = "muli16"} : index
    %61 = arith.addi %49, %60 {handshake.name = "addi16"} : index
    %62 = memref.load %arg6[%61] {handshake.deps = #handshake<deps[["store5", 3], ["store5", 4]]>, handshake.name = "load8"} : memref<100xi32>
    %63 = arith.addi %62, %59 {handshake.name = "addi2"} : i32
    %64 = arith.muli %48, %c10 {handshake.name = "muli17"} : index
    %65 = arith.addi %49, %64 {handshake.name = "addi17"} : index
    memref.store %63, %arg6[%65] {handshake.deps = #handshake<deps[["load8", 3], ["store5", 3]]>, handshake.name = "store5"} : memref<100xi32>
    %66 = arith.addi %52, %c1 {handshake.name = "addi24"} : index
    %67 = arith.cmpi ult, %66, %c10 {handshake.name = "cmpi6"} : index
    cf.cond_br %67, ^bb13(%66 : index), ^bb14 {handshake.name = "cond_br6"}
  ^bb14:  // pred: ^bb13
    %68 = arith.addi %49, %c1 {handshake.name = "addi25"} : index
    %69 = arith.cmpi ult, %68, %c10 {handshake.name = "cmpi7"} : index
    cf.cond_br %69, ^bb12(%68 : index), ^bb15 {handshake.name = "cond_br7"}
  ^bb15:  // pred: ^bb14
    %70 = arith.addi %48, %c1 {handshake.name = "addi26"} : index
    %71 = arith.cmpi ult, %70, %c10 {handshake.name = "cmpi8"} : index
    cf.cond_br %71, ^bb11(%70 : index), ^bb16 {handshake.name = "cond_br8"}
  ^bb16:  // pred: ^bb15
    return {handshake.name = "return0"}
  }
}

