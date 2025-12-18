module {
  func.func @kernel_3mm(%arg0: memref<100xi32> {handshake.arg_name = "A"}, %arg1: memref<100xi32> {handshake.arg_name = "B"}, %arg2: memref<100xi32> {handshake.arg_name = "C"}, %arg3: memref<100xi32> {handshake.arg_name = "D"}, %arg4: memref<100xi32> {handshake.arg_name = "E"}, %arg5: memref<100xi32> {handshake.arg_name = "F"}, %arg6: memref<100xi32> {handshake.arg_name = "G"}) {
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
    %4 = arith.addi %2, %3 {handshake.name = "addi27"} : index
    %5 = arith.addi %1, %4 {handshake.name = "addi3"} : index
    memref.store %c0_i32, %arg4[%5] {handshake.deps = #handshake<deps[["load20", 3, true], ["store13", 3, true], ["load24", 1, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store12"} : memref<100xi32>
    cf.br ^bb3(%c0_1 : index) {handshake.name = "br2"}
  ^bb3(%6: index):  // 2 preds: ^bb2, ^bb3
    %c10 = arith.constant {handshake.name = "constant10"} 10 : index
    %c1_2 = arith.constant {handshake.name = "constant11"} 1 : index
    %c3_3 = arith.constant {handshake.name = "constant12"} 3 : index
    %7 = arith.shli %0, %c1_2 {handshake.name = "shli2"} : index
    %8 = arith.shli %0, %c3_3 {handshake.name = "shli3"} : index
    %9 = arith.addi %7, %8 {handshake.name = "addi28"} : index
    %10 = arith.addi %6, %9 {handshake.name = "addi4"} : index
    %11 = memref.load %arg0[%10] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load18"} : memref<100xi32>
    %12 = arith.shli %6, %c1_2 {handshake.name = "shli4"} : index
    %13 = arith.shli %6, %c3_3 {handshake.name = "shli5"} : index
    %14 = arith.addi %12, %13 {handshake.name = "addi29"} : index
    %15 = arith.addi %1, %14 {handshake.name = "addi5"} : index
    %16 = memref.load %arg1[%15] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load19"} : memref<100xi32>
    %17 = arith.muli %11, %16 {handshake.name = "muli0"} : i32
    %18 = arith.shli %0, %c1_2 {handshake.name = "shli6"} : index
    %19 = arith.shli %0, %c3_3 {handshake.name = "shli7"} : index
    %20 = arith.addi %18, %19 {handshake.name = "addi30"} : index
    %21 = arith.addi %1, %20 {handshake.name = "addi6"} : index
    %22 = memref.load %arg4[%21] {handshake.deps = #handshake<deps[["store13", 3, true], ["store13", 4, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load20"} : memref<100xi32>
    %23 = arith.addi %22, %17 {handshake.name = "addi0"} : i32
    %24 = arith.shli %0, %c1_2 {handshake.name = "shli8"} : index
    %25 = arith.shli %0, %c3_3 {handshake.name = "shli9"} : index
    %26 = arith.addi %24, %25 {handshake.name = "addi31"} : index
    %27 = arith.addi %1, %26 {handshake.name = "addi7"} : index
    memref.store %23, %arg4[%27] {handshake.deps = #handshake<deps[["load20", 3, true], ["store13", 3, true], ["load24", 1, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store13"} : memref<100xi32>
    %28 = arith.addi %6, %c1_2 {handshake.name = "addi18"} : index
    %29 = arith.cmpi ult, %28, %c10 {handshake.name = "cmpi0"} : index
    cf.cond_br %29, ^bb3(%28 : index), ^bb4 {handshake.name = "cond_br0"}
  ^bb4:  // pred: ^bb3
    %c10_4 = arith.constant {handshake.name = "constant13"} 10 : index
    %c1_5 = arith.constant {handshake.name = "constant14"} 1 : index
    %30 = arith.addi %1, %c1_5 {handshake.name = "addi19"} : index
    %31 = arith.cmpi ult, %30, %c10_4 {handshake.name = "cmpi1"} : index
    cf.cond_br %31, ^bb2(%30 : index), ^bb5 {handshake.name = "cond_br1"}
  ^bb5:  // pred: ^bb4
    %c0_6 = arith.constant {handshake.name = "constant15"} 0 : index
    %c10_7 = arith.constant {handshake.name = "constant16"} 10 : index
    %c1_8 = arith.constant {handshake.name = "constant17"} 1 : index
    %32 = arith.addi %0, %c1_8 {handshake.name = "addi20"} : index
    %33 = arith.cmpi ult, %32, %c10_7 {handshake.name = "cmpi2"} : index
    cf.cond_br %33, ^bb1(%32 : index), ^bb6(%c0_6 : index) {handshake.name = "cond_br2"}
  ^bb6(%34: index):  // 2 preds: ^bb5, ^bb10
    %c0_9 = arith.constant {handshake.name = "constant18"} 0 : index
    cf.br ^bb7(%c0_9 : index) {handshake.name = "br4"}
  ^bb7(%35: index):  // 2 preds: ^bb6, ^bb9
    %c0_i32_10 = arith.constant {handshake.name = "constant19"} 0 : i32
    %c0_11 = arith.constant {handshake.name = "constant20"} 0 : index
    %c1_12 = arith.constant {handshake.name = "constant21"} 1 : index
    %c3_13 = arith.constant {handshake.name = "constant22"} 3 : index
    %36 = arith.shli %34, %c1_12 {handshake.name = "shli10"} : index
    %37 = arith.shli %34, %c3_13 {handshake.name = "shli11"} : index
    %38 = arith.addi %36, %37 {handshake.name = "addi32"} : index
    %39 = arith.addi %35, %38 {handshake.name = "addi8"} : index
    memref.store %c0_i32_10, %arg5[%39] {handshake.deps = #handshake<deps[["load23", 3, true], ["store15", 3, true], ["load25", 1, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store14"} : memref<100xi32>
    cf.br ^bb8(%c0_11 : index) {handshake.name = "br5"}
  ^bb8(%40: index):  // 2 preds: ^bb7, ^bb8
    %c10_14 = arith.constant {handshake.name = "constant23"} 10 : index
    %c1_15 = arith.constant {handshake.name = "constant24"} 1 : index
    %c3_16 = arith.constant {handshake.name = "constant25"} 3 : index
    %41 = arith.shli %34, %c1_15 {handshake.name = "shli12"} : index
    %42 = arith.shli %34, %c3_16 {handshake.name = "shli13"} : index
    %43 = arith.addi %41, %42 {handshake.name = "addi33"} : index
    %44 = arith.addi %40, %43 {handshake.name = "addi9"} : index
    %45 = memref.load %arg2[%44] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load21"} : memref<100xi32>
    %46 = arith.shli %40, %c1_15 {handshake.name = "shli14"} : index
    %47 = arith.shli %40, %c3_16 {handshake.name = "shli15"} : index
    %48 = arith.addi %46, %47 {handshake.name = "addi34"} : index
    %49 = arith.addi %35, %48 {handshake.name = "addi10"} : index
    %50 = memref.load %arg3[%49] {handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load22"} : memref<100xi32>
    %51 = arith.muli %45, %50 {handshake.name = "muli1"} : i32
    %52 = arith.shli %34, %c1_15 {handshake.name = "shli16"} : index
    %53 = arith.shli %34, %c3_16 {handshake.name = "shli17"} : index
    %54 = arith.addi %52, %53 {handshake.name = "addi35"} : index
    %55 = arith.addi %35, %54 {handshake.name = "addi11"} : index
    %56 = memref.load %arg5[%55] {handshake.deps = #handshake<deps[["store15", 3, true], ["store15", 4, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load23"} : memref<100xi32>
    %57 = arith.addi %56, %51 {handshake.name = "addi1"} : i32
    %58 = arith.shli %34, %c1_15 {handshake.name = "shli18"} : index
    %59 = arith.shli %34, %c3_16 {handshake.name = "shli19"} : index
    %60 = arith.addi %58, %59 {handshake.name = "addi36"} : index
    %61 = arith.addi %35, %60 {handshake.name = "addi12"} : index
    memref.store %57, %arg5[%61] {handshake.deps = #handshake<deps[["load23", 3, true], ["store15", 3, true], ["load25", 1, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store15"} : memref<100xi32>
    %62 = arith.addi %40, %c1_15 {handshake.name = "addi21"} : index
    %63 = arith.cmpi ult, %62, %c10_14 {handshake.name = "cmpi3"} : index
    cf.cond_br %63, ^bb8(%62 : index), ^bb9 {handshake.name = "cond_br3"}
  ^bb9:  // pred: ^bb8
    %c10_17 = arith.constant {handshake.name = "constant26"} 10 : index
    %c1_18 = arith.constant {handshake.name = "constant27"} 1 : index
    %64 = arith.addi %35, %c1_18 {handshake.name = "addi22"} : index
    %65 = arith.cmpi ult, %64, %c10_17 {handshake.name = "cmpi4"} : index
    cf.cond_br %65, ^bb7(%64 : index), ^bb10 {handshake.name = "cond_br4"}
  ^bb10:  // pred: ^bb9
    %c0_19 = arith.constant {handshake.name = "constant28"} 0 : index
    %c10_20 = arith.constant {handshake.name = "constant29"} 10 : index
    %c1_21 = arith.constant {handshake.name = "constant30"} 1 : index
    %66 = arith.addi %34, %c1_21 {handshake.name = "addi23"} : index
    %67 = arith.cmpi ult, %66, %c10_20 {handshake.name = "cmpi5"} : index
    cf.cond_br %67, ^bb6(%66 : index), ^bb11(%c0_19 : index) {handshake.name = "cond_br5"}
  ^bb11(%68: index):  // 2 preds: ^bb10, ^bb15
    %c0_22 = arith.constant {handshake.name = "constant31"} 0 : index
    cf.br ^bb12(%c0_22 : index) {handshake.name = "br7"}
  ^bb12(%69: index):  // 2 preds: ^bb11, ^bb14
    %c0_i32_23 = arith.constant {handshake.name = "constant32"} 0 : i32
    %c0_24 = arith.constant {handshake.name = "constant33"} 0 : index
    %c1_25 = arith.constant {handshake.name = "constant34"} 1 : index
    %c3_26 = arith.constant {handshake.name = "constant35"} 3 : index
    %70 = arith.shli %68, %c1_25 {handshake.name = "shli20"} : index
    %71 = arith.shli %68, %c3_26 {handshake.name = "shli21"} : index
    %72 = arith.addi %70, %71 {handshake.name = "addi37"} : index
    %73 = arith.addi %69, %72 {handshake.name = "addi13"} : index
    memref.store %c0_i32_23, %arg6[%73] {handshake.deps = #handshake<deps[["load26", 3, true], ["store17", 3, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store16"} : memref<100xi32>
    cf.br ^bb13(%c0_24 : index) {handshake.name = "br8"}
  ^bb13(%74: index):  // 2 preds: ^bb12, ^bb13
    %c10_27 = arith.constant {handshake.name = "constant36"} 10 : index
    %c1_28 = arith.constant {handshake.name = "constant37"} 1 : index
    %c3_29 = arith.constant {handshake.name = "constant38"} 3 : index
    %75 = arith.shli %68, %c1_28 {handshake.name = "shli22"} : index
    %76 = arith.shli %68, %c3_29 {handshake.name = "shli23"} : index
    %77 = arith.addi %75, %76 {handshake.name = "addi38"} : index
    %78 = arith.addi %74, %77 {handshake.name = "addi14"} : index
    %79 = memref.load %arg4[%78] {handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load24"} : memref<100xi32>
    %80 = arith.shli %74, %c1_28 {handshake.name = "shli24"} : index
    %81 = arith.shli %74, %c3_29 {handshake.name = "shli25"} : index
    %82 = arith.addi %80, %81 {handshake.name = "addi39"} : index
    %83 = arith.addi %69, %82 {handshake.name = "addi15"} : index
    %84 = memref.load %arg5[%83] {handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load25"} : memref<100xi32>
    %85 = arith.muli %79, %84 {handshake.name = "muli2"} : i32
    %86 = arith.shli %68, %c1_28 {handshake.name = "shli26"} : index
    %87 = arith.shli %68, %c3_29 {handshake.name = "shli27"} : index
    %88 = arith.addi %86, %87 {handshake.name = "addi40"} : index
    %89 = arith.addi %69, %88 {handshake.name = "addi16"} : index
    %90 = memref.load %arg6[%89] {handshake.deps = #handshake<deps[["store17", 3, true], ["store17", 4, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load26"} : memref<100xi32>
    %91 = arith.addi %90, %85 {handshake.name = "addi2"} : i32
    %92 = arith.shli %68, %c1_28 {handshake.name = "shli28"} : index
    %93 = arith.shli %68, %c3_29 {handshake.name = "shli29"} : index
    %94 = arith.addi %92, %93 {handshake.name = "addi41"} : index
    %95 = arith.addi %69, %94 {handshake.name = "addi17"} : index
    memref.store %91, %arg6[%95] {handshake.deps = #handshake<deps[["load26", 3, true], ["store17", 3, true]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store17"} : memref<100xi32>
    %96 = arith.addi %74, %c1_28 {handshake.name = "addi24"} : index
    %97 = arith.cmpi ult, %96, %c10_27 {handshake.name = "cmpi6"} : index
    cf.cond_br %97, ^bb13(%96 : index), ^bb14 {handshake.name = "cond_br6"}
  ^bb14:  // pred: ^bb13
    %c10_30 = arith.constant {handshake.name = "constant39"} 10 : index
    %c1_31 = arith.constant {handshake.name = "constant40"} 1 : index
    %98 = arith.addi %69, %c1_31 {handshake.name = "addi25"} : index
    %99 = arith.cmpi ult, %98, %c10_30 {handshake.name = "cmpi7"} : index
    cf.cond_br %99, ^bb12(%98 : index), ^bb15 {handshake.name = "cond_br7"}
  ^bb15:  // pred: ^bb14
    %c10_32 = arith.constant {handshake.name = "constant41"} 10 : index
    %c1_33 = arith.constant {handshake.name = "constant42"} 1 : index
    %100 = arith.addi %68, %c1_33 {handshake.name = "addi26"} : index
    %101 = arith.cmpi ult, %100, %c10_32 {handshake.name = "cmpi8"} : index
    cf.cond_br %101, ^bb11(%100 : index), ^bb16 {handshake.name = "cond_br8"}
  ^bb16:  // pred: ^bb15
    return {handshake.name = "return0"}
  }
}

