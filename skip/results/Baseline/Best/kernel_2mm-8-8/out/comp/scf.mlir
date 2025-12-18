module {
  func.func @kernel_2mm(%arg0: i32 {handshake.arg_name = "alpha"}, %arg1: i32 {handshake.arg_name = "beta"}, %arg2: memref<100xi32> {handshake.arg_name = "tmp"}, %arg3: memref<100xi32> {handshake.arg_name = "A"}, %arg4: memref<100xi32> {handshake.arg_name = "B"}, %arg5: memref<100xi32> {handshake.arg_name = "C"}, %arg6: memref<100xi32> {handshake.arg_name = "D"}) {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %c10 = arith.constant {handshake.name = "constant2"} 10 : index
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    %0 = scf.while (%arg7 = %c0) : (index) -> index {
      %2 = scf.while (%arg8 = %c0) : (index) -> index {
        %5 = arith.muli %arg7, %c10 {handshake.name = "muli4"} : index
        %6 = arith.addi %arg8, %5 {handshake.name = "addi2"} : index
        memref.store %c0_i32, %arg2[%6] {handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.name = "store0"} : memref<100xi32>
        %7 = scf.while (%arg9 = %c0) : (index) -> index {
          %10 = arith.muli %arg7, %c10 {handshake.name = "muli5"} : index
          %11 = arith.addi %arg9, %10 {handshake.name = "addi3"} : index
          %12 = memref.load %arg3[%11] {handshake.name = "load0"} : memref<100xi32>
          %13 = arith.muli %arg0, %12 {handshake.name = "muli0"} : i32
          %14 = arith.muli %arg9, %c10 {handshake.name = "muli6"} : index
          %15 = arith.addi %arg8, %14 {handshake.name = "addi4"} : index
          %16 = memref.load %arg4[%15] {handshake.name = "load1"} : memref<100xi32>
          %17 = arith.muli %13, %16 {handshake.name = "muli1"} : i32
          %18 = arith.muli %arg7, %c10 {handshake.name = "muli7"} : index
          %19 = arith.addi %arg8, %18 {handshake.name = "addi5"} : index
          %20 = memref.load %arg2[%19] {handshake.deps = #handshake<deps[["store1", 3], ["store1", 4]]>, handshake.name = "load2"} : memref<100xi32>
          %21 = arith.addi %20, %17 {handshake.name = "addi0"} : i32
          %22 = arith.muli %arg7, %c10 {handshake.name = "muli8"} : index
          %23 = arith.addi %arg8, %22 {handshake.name = "addi6"} : index
          memref.store %21, %arg2[%23] {handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.name = "store1"} : memref<100xi32>
          %24 = arith.addi %arg9, %c1 {handshake.name = "addi13"} : index
          %25 = arith.cmpi ult, %24, %c10 {handshake.name = "cmpi0"} : index
          scf.condition(%25) {handshake.name = "condition0"} %24 : index
        } do {
        ^bb0(%arg9: index):
          scf.yield {handshake.name = "yield6"} %arg9 : index
        } attributes {handshake.name = "while0"}
        %8 = arith.addi %arg8, %c1 {handshake.name = "addi14"} : index
        %9 = arith.cmpi ult, %8, %c10 {handshake.name = "cmpi1"} : index
        scf.condition(%9) {handshake.name = "condition1"} %8 : index
      } do {
      ^bb0(%arg8: index):
        scf.yield {handshake.name = "yield7"} %arg8 : index
      } attributes {handshake.name = "while1"}
      %3 = arith.addi %arg7, %c1 {handshake.name = "addi15"} : index
      %4 = arith.cmpi ult, %3, %c10 {handshake.name = "cmpi2"} : index
      scf.condition(%4) {handshake.name = "condition2"} %3 : index
    } do {
    ^bb0(%arg7: index):
      scf.yield {handshake.name = "yield8"} %arg7 : index
    } attributes {handshake.name = "while2"}
    %1 = scf.while (%arg7 = %c0) : (index) -> index {
      %2 = scf.while (%arg8 = %c0) : (index) -> index {
        %5 = arith.muli %arg7, %c10 {handshake.name = "muli9"} : index
        %6 = arith.addi %arg8, %5 {handshake.name = "addi7"} : index
        %7 = memref.load %arg6[%6] {handshake.deps = #handshake<deps[["store2", 3], ["store3", 3]]>, handshake.name = "load3"} : memref<100xi32>
        %8 = arith.muli %7, %arg1 {handshake.name = "muli2"} : i32
        %9 = arith.muli %arg7, %c10 {handshake.name = "muli10"} : index
        %10 = arith.addi %arg8, %9 {handshake.name = "addi8"} : index
        memref.store %8, %arg6[%10] {handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.name = "store2"} : memref<100xi32>
        %11 = scf.while (%arg9 = %c0) : (index) -> index {
          %14 = arith.muli %arg7, %c10 {handshake.name = "muli11"} : index
          %15 = arith.addi %arg9, %14 {handshake.name = "addi9"} : index
          %16 = memref.load %arg2[%15] {handshake.name = "load4"} : memref<100xi32>
          %17 = arith.muli %arg9, %c10 {handshake.name = "muli12"} : index
          %18 = arith.addi %arg8, %17 {handshake.name = "addi10"} : index
          %19 = memref.load %arg5[%18] {handshake.name = "load5"} : memref<100xi32>
          %20 = arith.muli %16, %19 {handshake.name = "muli3"} : i32
          %21 = arith.muli %arg7, %c10 {handshake.name = "muli13"} : index
          %22 = arith.addi %arg8, %21 {handshake.name = "addi11"} : index
          %23 = memref.load %arg6[%22] {handshake.deps = #handshake<deps[["store3", 3], ["store3", 4]]>, handshake.name = "load6"} : memref<100xi32>
          %24 = arith.addi %23, %20 {handshake.name = "addi1"} : i32
          %25 = arith.muli %arg7, %c10 {handshake.name = "muli14"} : index
          %26 = arith.addi %arg8, %25 {handshake.name = "addi12"} : index
          memref.store %24, %arg6[%26] {handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.name = "store3"} : memref<100xi32>
          %27 = arith.addi %arg9, %c1 {handshake.name = "addi16"} : index
          %28 = arith.cmpi ult, %27, %c10 {handshake.name = "cmpi3"} : index
          scf.condition(%28) {handshake.name = "condition3"} %27 : index
        } do {
        ^bb0(%arg9: index):
          scf.yield {handshake.name = "yield9"} %arg9 : index
        } attributes {handshake.name = "while3"}
        %12 = arith.addi %arg8, %c1 {handshake.name = "addi17"} : index
        %13 = arith.cmpi ult, %12, %c10 {handshake.name = "cmpi4"} : index
        scf.condition(%13) {handshake.name = "condition4"} %12 : index
      } do {
      ^bb0(%arg8: index):
        scf.yield {handshake.name = "yield10"} %arg8 : index
      } attributes {handshake.name = "while4"}
      %3 = arith.addi %arg7, %c1 {handshake.name = "addi18"} : index
      %4 = arith.cmpi ult, %3, %c10 {handshake.name = "cmpi5"} : index
      scf.condition(%4) {handshake.name = "condition5"} %3 : index
    } do {
    ^bb0(%arg7: index):
      scf.yield {handshake.name = "yield11"} %arg7 : index
    } attributes {handshake.name = "while5"}
    return {handshake.name = "return0"}
  }
}

