module {
  func.func @kernel_3mm(%arg0: memref<100xi32> {handshake.arg_name = "A"}, %arg1: memref<100xi32> {handshake.arg_name = "B"}, %arg2: memref<100xi32> {handshake.arg_name = "C"}, %arg3: memref<100xi32> {handshake.arg_name = "D"}, %arg4: memref<100xi32> {handshake.arg_name = "E"}, %arg5: memref<100xi32> {handshake.arg_name = "F"}, %arg6: memref<100xi32> {handshake.arg_name = "G"}) {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %c10 = arith.constant {handshake.name = "constant2"} 10 : index
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    %0 = scf.while (%arg7 = %c0) : (index) -> index {
      %3 = scf.while (%arg8 = %c0) : (index) -> index {
        %6 = arith.muli %arg7, %c10 {handshake.name = "muli3"} : index
        %7 = arith.addi %arg8, %6 {handshake.name = "addi3"} : index
        memref.store %c0_i32, %arg4[%7] {handshake.deps = #handshake<deps[["load20", 3, true], ["store13", 3, true], ["load24", 1, true]]>, handshake.name = "store12"} : memref<100xi32>
        %8 = scf.while (%arg9 = %c0) : (index) -> index {
          %11 = arith.muli %arg7, %c10 {handshake.name = "muli4"} : index
          %12 = arith.addi %arg9, %11 {handshake.name = "addi4"} : index
          %13 = memref.load %arg0[%12] {handshake.name = "load18"} : memref<100xi32>
          %14 = arith.muli %arg9, %c10 {handshake.name = "muli5"} : index
          %15 = arith.addi %arg8, %14 {handshake.name = "addi5"} : index
          %16 = memref.load %arg1[%15] {handshake.name = "load19"} : memref<100xi32>
          %17 = arith.muli %13, %16 {handshake.name = "muli0"} : i32
          %18 = arith.muli %arg7, %c10 {handshake.name = "muli6"} : index
          %19 = arith.addi %arg8, %18 {handshake.name = "addi6"} : index
          %20 = memref.load %arg4[%19] {handshake.deps = #handshake<deps[["store13", 3, true], ["store13", 4, true]]>, handshake.name = "load20"} : memref<100xi32>
          %21 = arith.addi %20, %17 {handshake.name = "addi0"} : i32
          %22 = arith.muli %arg7, %c10 {handshake.name = "muli7"} : index
          %23 = arith.addi %arg8, %22 {handshake.name = "addi7"} : index
          memref.store %21, %arg4[%23] {handshake.deps = #handshake<deps[["load20", 3, true], ["store13", 3, true], ["load24", 1, true]]>, handshake.name = "store13"} : memref<100xi32>
          %24 = arith.addi %arg9, %c1 {handshake.name = "addi18"} : index
          %25 = arith.cmpi ult, %24, %c10 {handshake.name = "cmpi0"} : index
          scf.condition(%25) {handshake.name = "condition0"} %24 : index
        } do {
        ^bb0(%arg9: index):
          scf.yield {handshake.name = "yield9"} %arg9 : index
        } attributes {handshake.name = "while0"}
        %9 = arith.addi %arg8, %c1 {handshake.name = "addi19"} : index
        %10 = arith.cmpi ult, %9, %c10 {handshake.name = "cmpi1"} : index
        scf.condition(%10) {handshake.name = "condition1"} %9 : index
      } do {
      ^bb0(%arg8: index):
        scf.yield {handshake.name = "yield10"} %arg8 : index
      } attributes {handshake.name = "while1"}
      %4 = arith.addi %arg7, %c1 {handshake.name = "addi20"} : index
      %5 = arith.cmpi ult, %4, %c10 {handshake.name = "cmpi2"} : index
      scf.condition(%5) {handshake.name = "condition2"} %4 : index
    } do {
    ^bb0(%arg7: index):
      scf.yield {handshake.name = "yield11"} %arg7 : index
    } attributes {handshake.name = "while2"}
    %1 = scf.while (%arg7 = %c0) : (index) -> index {
      %3 = scf.while (%arg8 = %c0) : (index) -> index {
        %6 = arith.muli %arg7, %c10 {handshake.name = "muli8"} : index
        %7 = arith.addi %arg8, %6 {handshake.name = "addi8"} : index
        memref.store %c0_i32, %arg5[%7] {handshake.deps = #handshake<deps[["load23", 3, true], ["store15", 3, true], ["load25", 1, true]]>, handshake.name = "store14"} : memref<100xi32>
        %8 = scf.while (%arg9 = %c0) : (index) -> index {
          %11 = arith.muli %arg7, %c10 {handshake.name = "muli9"} : index
          %12 = arith.addi %arg9, %11 {handshake.name = "addi9"} : index
          %13 = memref.load %arg2[%12] {handshake.name = "load21"} : memref<100xi32>
          %14 = arith.muli %arg9, %c10 {handshake.name = "muli10"} : index
          %15 = arith.addi %arg8, %14 {handshake.name = "addi10"} : index
          %16 = memref.load %arg3[%15] {handshake.name = "load22"} : memref<100xi32>
          %17 = arith.muli %13, %16 {handshake.name = "muli1"} : i32
          %18 = arith.muli %arg7, %c10 {handshake.name = "muli11"} : index
          %19 = arith.addi %arg8, %18 {handshake.name = "addi11"} : index
          %20 = memref.load %arg5[%19] {handshake.deps = #handshake<deps[["store15", 3, true], ["store15", 4, true]]>, handshake.name = "load23"} : memref<100xi32>
          %21 = arith.addi %20, %17 {handshake.name = "addi1"} : i32
          %22 = arith.muli %arg7, %c10 {handshake.name = "muli12"} : index
          %23 = arith.addi %arg8, %22 {handshake.name = "addi12"} : index
          memref.store %21, %arg5[%23] {handshake.deps = #handshake<deps[["load23", 3, true], ["store15", 3, true], ["load25", 1, true]]>, handshake.name = "store15"} : memref<100xi32>
          %24 = arith.addi %arg9, %c1 {handshake.name = "addi21"} : index
          %25 = arith.cmpi ult, %24, %c10 {handshake.name = "cmpi3"} : index
          scf.condition(%25) {handshake.name = "condition3"} %24 : index
        } do {
        ^bb0(%arg9: index):
          scf.yield {handshake.name = "yield12"} %arg9 : index
        } attributes {handshake.name = "while3"}
        %9 = arith.addi %arg8, %c1 {handshake.name = "addi22"} : index
        %10 = arith.cmpi ult, %9, %c10 {handshake.name = "cmpi4"} : index
        scf.condition(%10) {handshake.name = "condition4"} %9 : index
      } do {
      ^bb0(%arg8: index):
        scf.yield {handshake.name = "yield13"} %arg8 : index
      } attributes {handshake.name = "while4"}
      %4 = arith.addi %arg7, %c1 {handshake.name = "addi23"} : index
      %5 = arith.cmpi ult, %4, %c10 {handshake.name = "cmpi5"} : index
      scf.condition(%5) {handshake.name = "condition5"} %4 : index
    } do {
    ^bb0(%arg7: index):
      scf.yield {handshake.name = "yield14"} %arg7 : index
    } attributes {handshake.name = "while5"}
    %2 = scf.while (%arg7 = %c0) : (index) -> index {
      %3 = scf.while (%arg8 = %c0) : (index) -> index {
        %6 = arith.muli %arg7, %c10 {handshake.name = "muli13"} : index
        %7 = arith.addi %arg8, %6 {handshake.name = "addi13"} : index
        memref.store %c0_i32, %arg6[%7] {handshake.deps = #handshake<deps[["load26", 3, true], ["store17", 3, true]]>, handshake.name = "store16"} : memref<100xi32>
        %8 = scf.while (%arg9 = %c0) : (index) -> index {
          %11 = arith.muli %arg7, %c10 {handshake.name = "muli14"} : index
          %12 = arith.addi %arg9, %11 {handshake.name = "addi14"} : index
          %13 = memref.load %arg4[%12] {handshake.name = "load24"} : memref<100xi32>
          %14 = arith.muli %arg9, %c10 {handshake.name = "muli15"} : index
          %15 = arith.addi %arg8, %14 {handshake.name = "addi15"} : index
          %16 = memref.load %arg5[%15] {handshake.name = "load25"} : memref<100xi32>
          %17 = arith.muli %13, %16 {handshake.name = "muli2"} : i32
          %18 = arith.muli %arg7, %c10 {handshake.name = "muli16"} : index
          %19 = arith.addi %arg8, %18 {handshake.name = "addi16"} : index
          %20 = memref.load %arg6[%19] {handshake.deps = #handshake<deps[["store17", 3, true], ["store17", 4, true]]>, handshake.name = "load26"} : memref<100xi32>
          %21 = arith.addi %20, %17 {handshake.name = "addi2"} : i32
          %22 = arith.muli %arg7, %c10 {handshake.name = "muli17"} : index
          %23 = arith.addi %arg8, %22 {handshake.name = "addi17"} : index
          memref.store %21, %arg6[%23] {handshake.deps = #handshake<deps[["load26", 3, true], ["store17", 3, true]]>, handshake.name = "store17"} : memref<100xi32>
          %24 = arith.addi %arg9, %c1 {handshake.name = "addi24"} : index
          %25 = arith.cmpi ult, %24, %c10 {handshake.name = "cmpi6"} : index
          scf.condition(%25) {handshake.name = "condition6"} %24 : index
        } do {
        ^bb0(%arg9: index):
          scf.yield {handshake.name = "yield15"} %arg9 : index
        } attributes {handshake.name = "while6"}
        %9 = arith.addi %arg8, %c1 {handshake.name = "addi25"} : index
        %10 = arith.cmpi ult, %9, %c10 {handshake.name = "cmpi7"} : index
        scf.condition(%10) {handshake.name = "condition7"} %9 : index
      } do {
      ^bb0(%arg8: index):
        scf.yield {handshake.name = "yield16"} %arg8 : index
      } attributes {handshake.name = "while7"}
      %4 = arith.addi %arg7, %c1 {handshake.name = "addi26"} : index
      %5 = arith.cmpi ult, %4, %c10 {handshake.name = "cmpi8"} : index
      scf.condition(%5) {handshake.name = "condition8"} %4 : index
    } do {
    ^bb0(%arg7: index):
      scf.yield {handshake.name = "yield17"} %arg7 : index
    } attributes {handshake.name = "while8"}
    return {handshake.name = "return0"}
  }
}

