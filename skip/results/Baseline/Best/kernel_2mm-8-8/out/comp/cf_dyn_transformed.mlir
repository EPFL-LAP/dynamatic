module {
  func.func @kernel_2mm(%arg0: i32 {handshake.arg_name = "alpha"}, %arg1: i32 {handshake.arg_name = "beta"}, %arg2: memref<100xi32> {handshake.arg_name = "tmp"}, %arg3: memref<100xi32> {handshake.arg_name = "A"}, %arg4: memref<100xi32> {handshake.arg_name = "B"}, %arg5: memref<100xi32> {handshake.arg_name = "C"}, %arg6: memref<100xi32> {handshake.arg_name = "D"}) {
    %c0 = arith.constant {handshake.name = "constant3"} 0 : index
    cf.br ^bb1(%c0 : index) {handshake.name = "br0"}
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    %c0_0 = arith.constant {handshake.name = "constant5"} 0 : index
    cf.br ^bb2(%c0_0 : index) {handshake.name = "br1"}
  ^bb2(%1: index):  // 2 preds: ^bb1, ^bb4
    %c0_i32 = arith.constant {handshake.name = "constant6"} 0 : i32
    %c0_1 = arith.constant {handshake.name = "constant7"} 0 : index
    %c1 = arith.constant {handshake.name = "constant8"} 1 : index
    %c3 = arith.constant {handshake.name = "constant9"} 3 : index
    %2 = arith.shli %0, %c1 {handshake.name = "shli0"} : index
    %3 = arith.shli %0, %c3 {handshake.name = "shli1"} : index
    %4 = arith.addi %2, %3 {handshake.name = "addi19"} : index
    %5 = arith.addi %1, %4 {handshake.name = "addi2"} : index
    memref.store %c0_i32, %arg2[%5] {handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.name = "store0"} : memref<100xi32>
    cf.br ^bb3(%c0_1 : index) {handshake.name = "br2"}
  ^bb3(%6: index):  // 2 preds: ^bb2, ^bb3
    %c10 = arith.constant {handshake.name = "constant10"} 10 : index
    %c1_2 = arith.constant {handshake.name = "constant11"} 1 : index
    %c3_3 = arith.constant {handshake.name = "constant12"} 3 : index
    %7 = arith.shli %0, %c1_2 {handshake.name = "shli2"} : index
    %8 = arith.shli %0, %c3_3 {handshake.name = "shli3"} : index
    %9 = arith.addi %7, %8 {handshake.name = "addi20"} : index
    %10 = arith.addi %6, %9 {handshake.name = "addi3"} : index
    %11 = memref.load %arg3[%10] {handshake.name = "load0"} : memref<100xi32>
    %12 = arith.muli %arg0, %11 {handshake.name = "muli0"} : i32
    %13 = arith.shli %6, %c1_2 {handshake.name = "shli4"} : index
    %14 = arith.shli %6, %c3_3 {handshake.name = "shli5"} : index
    %15 = arith.addi %13, %14 {handshake.name = "addi21"} : index
    %16 = arith.addi %1, %15 {handshake.name = "addi4"} : index
    %17 = memref.load %arg4[%16] {handshake.name = "load1"} : memref<100xi32>
    %18 = arith.muli %12, %17 {handshake.name = "muli1"} : i32
    %19 = arith.shli %0, %c1_2 {handshake.name = "shli6"} : index
    %20 = arith.shli %0, %c3_3 {handshake.name = "shli7"} : index
    %21 = arith.addi %19, %20 {handshake.name = "addi22"} : index
    %22 = arith.addi %1, %21 {handshake.name = "addi5"} : index
    %23 = memref.load %arg2[%22] {handshake.deps = #handshake<deps[["store1", 3], ["store1", 4]]>, handshake.name = "load2"} : memref<100xi32>
    %24 = arith.addi %23, %18 {handshake.name = "addi0"} : i32
    %25 = arith.shli %0, %c1_2 {handshake.name = "shli8"} : index
    %26 = arith.shli %0, %c3_3 {handshake.name = "shli9"} : index
    %27 = arith.addi %25, %26 {handshake.name = "addi23"} : index
    %28 = arith.addi %1, %27 {handshake.name = "addi6"} : index
    memref.store %24, %arg2[%28] {handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.name = "store1"} : memref<100xi32>
    %29 = arith.addi %6, %c1_2 {handshake.name = "addi13"} : index
    %30 = arith.cmpi ult, %29, %c10 {handshake.name = "cmpi0"} : index
    cf.cond_br %30, ^bb3(%29 : index), ^bb4 {handshake.name = "cond_br0"}
  ^bb4:  // pred: ^bb3
    %c10_4 = arith.constant {handshake.name = "constant13"} 10 : index
    %c1_5 = arith.constant {handshake.name = "constant14"} 1 : index
    %31 = arith.addi %1, %c1_5 {handshake.name = "addi14"} : index
    %32 = arith.cmpi ult, %31, %c10_4 {handshake.name = "cmpi1"} : index
    cf.cond_br %32, ^bb2(%31 : index), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb4
    %c0_6 = arith.constant {handshake.name = "constant15"} 0 : index
    %c10_7 = arith.constant {handshake.name = "constant16"} 10 : index
    %c1_8 = arith.constant {handshake.name = "constant17"} 1 : index
    %33 = arith.addi %0, %c1_8 {handshake.name = "addi15"} : index
    %34 = arith.cmpi ult, %33, %c10_7 {handshake.name = "cmpi2"} : index
    cf.cond_br %34, ^bb1(%33 : index), ^bb6(%c0_6 : index) {handshake.name = "cond_br2"}
  ^bb6(%35: index):  // 2 preds: ^bb5, ^bb10
    %c0_9 = arith.constant {handshake.name = "constant18"} 0 : index
    cf.br ^bb7(%c0_9 : index) {handshake.name = "br4"}
  ^bb7(%36: index):  // 2 preds: ^bb6, ^bb9
    %c0_10 = arith.constant {handshake.name = "constant19"} 0 : index
    %c1_11 = arith.constant {handshake.name = "constant20"} 1 : index
    %c3_12 = arith.constant {handshake.name = "constant21"} 3 : index
    %37 = arith.shli %35, %c1_11 {handshake.name = "shli10"} : index
    %38 = arith.shli %35, %c3_12 {handshake.name = "shli11"} : index
    %39 = arith.addi %37, %38 {handshake.name = "addi24"} : index
    %40 = arith.addi %36, %39 {handshake.name = "addi7"} : index
    %41 = memref.load %arg6[%40] {handshake.deps = #handshake<deps[["store2", 3], ["store3", 3]]>, handshake.name = "load3"} : memref<100xi32>
    %42 = arith.muli %41, %arg1 {handshake.name = "muli2"} : i32
    %43 = arith.shli %35, %c1_11 {handshake.name = "shli12"} : index
    %44 = arith.shli %35, %c3_12 {handshake.name = "shli13"} : index
    %45 = arith.addi %43, %44 {handshake.name = "addi25"} : index
    %46 = arith.addi %36, %45 {handshake.name = "addi8"} : index
    memref.store %42, %arg6[%46] {handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.name = "store2"} : memref<100xi32>
    cf.br ^bb8(%c0_10 : index) {handshake.name = "br5"}
  ^bb8(%47: index):  // 2 preds: ^bb7, ^bb8
    %c10_13 = arith.constant {handshake.name = "constant22"} 10 : index
    %c1_14 = arith.constant {handshake.name = "constant23"} 1 : index
    %c3_15 = arith.constant {handshake.name = "constant24"} 3 : index
    %48 = arith.shli %35, %c1_14 {handshake.name = "shli14"} : index
    %49 = arith.shli %35, %c3_15 {handshake.name = "shli15"} : index
    %50 = arith.addi %48, %49 {handshake.name = "addi26"} : index
    %51 = arith.addi %47, %50 {handshake.name = "addi9"} : index
    %52 = memref.load %arg2[%51] {handshake.name = "load4"} : memref<100xi32>
    %53 = arith.shli %47, %c1_14 {handshake.name = "shli16"} : index
    %54 = arith.shli %47, %c3_15 {handshake.name = "shli17"} : index
    %55 = arith.addi %53, %54 {handshake.name = "addi27"} : index
    %56 = arith.addi %36, %55 {handshake.name = "addi10"} : index
    %57 = memref.load %arg5[%56] {handshake.name = "load5"} : memref<100xi32>
    %58 = arith.muli %52, %57 {handshake.name = "muli3"} : i32
    %59 = arith.shli %35, %c1_14 {handshake.name = "shli18"} : index
    %60 = arith.shli %35, %c3_15 {handshake.name = "shli19"} : index
    %61 = arith.addi %59, %60 {handshake.name = "addi28"} : index
    %62 = arith.addi %36, %61 {handshake.name = "addi11"} : index
    %63 = memref.load %arg6[%62] {handshake.deps = #handshake<deps[["store3", 3], ["store3", 4]]>, handshake.name = "load6"} : memref<100xi32>
    %64 = arith.addi %63, %58 {handshake.name = "addi1"} : i32
    %65 = arith.shli %35, %c1_14 {handshake.name = "shli20"} : index
    %66 = arith.shli %35, %c3_15 {handshake.name = "shli21"} : index
    %67 = arith.addi %65, %66 {handshake.name = "addi29"} : index
    %68 = arith.addi %36, %67 {handshake.name = "addi12"} : index
    memref.store %64, %arg6[%68] {handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.name = "store3"} : memref<100xi32>
    %69 = arith.addi %47, %c1_14 {handshake.name = "addi16"} : index
    %70 = arith.cmpi ult, %69, %c10_13 {handshake.name = "cmpi3"} : index
    cf.cond_br %70, ^bb8(%69 : index), ^bb9 {handshake.name = "cond_br3"}
  ^bb9:  // pred: ^bb8
    %c10_16 = arith.constant {handshake.name = "constant25"} 10 : index
    %c1_17 = arith.constant {handshake.name = "constant26"} 1 : index
    %71 = arith.addi %36, %c1_17 {handshake.name = "addi17"} : index
    %72 = arith.cmpi ult, %71, %c10_16 {handshake.name = "cmpi4"} : index
    cf.cond_br %72, ^bb7(%71 : index), ^bb10 {handshake.name = "cond_br4"}
  ^bb10:  // pred: ^bb9
    %c10_18 = arith.constant {handshake.name = "constant27"} 10 : index
    %c1_19 = arith.constant {handshake.name = "constant28"} 1 : index
    %73 = arith.addi %35, %c1_19 {handshake.name = "addi18"} : index
    %74 = arith.cmpi ult, %73, %c10_18 {handshake.name = "cmpi5"} : index
    cf.cond_br %74, ^bb6(%73 : index), ^bb11 {handshake.name = "cond_br5"}
  ^bb11:  // pred: ^bb10
    return {handshake.name = "return0"}
  }
}

