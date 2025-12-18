module {
  func.func @matrix_power(%arg0: memref<400xi32> {handshake.arg_name = "mat"}, %arg1: memref<20xi32> {handshake.arg_name = "row"}, %arg2: memref<20xi32> {handshake.arg_name = "col"}, %arg3: memref<20xi32> {handshake.arg_name = "a"}) {
    %c0 = arith.constant {handshake.name = "constant4"} 0 : index
    %c-1_i32 = arith.constant {handshake.name = "constant0"} -1 : i32
    %c1 = arith.constant {handshake.name = "constant1"} 1 : index
    %c20 = arith.constant {handshake.name = "constant2"} 20 : index
    %0 = scf.while (%arg4 = %c1) : (index) -> index {
      %1 = arith.index_cast %arg4 {handshake.name = "index_cast0"} : index to i32
      %2 = arith.addi %1, %c-1_i32 {handshake.name = "addi0"} : i32
      %3 = arith.index_cast %2 {handshake.name = "index_cast1"} : i32 to index
      %4 = scf.while (%arg5 = %c0) : (index) -> index {
        %7 = memref.load %arg1[%arg5] {handshake.name = "load5"} : memref<20xi32>
        %8 = arith.index_cast %7 {handshake.name = "index_cast2"} : i32 to index
        %9 = memref.load %arg3[%arg5] {handshake.name = "load6"} : memref<20xi32>
        %10 = memref.load %arg2[%arg5] {handshake.name = "load7"} : memref<20xi32>
        %11 = arith.index_cast %10 {handshake.name = "index_cast3"} : i32 to index
        %12 = arith.muli %3, %c20 {handshake.name = "muli1"} : index
        %13 = arith.addi %11, %12 {handshake.name = "addi2"} : index
        %14 = memref.load %arg0[%13] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.name = "load8"} : memref<400xi32>
        %15 = arith.muli %9, %14 {handshake.name = "muli0"} : i32
        %16 = arith.muli %arg4, %c20 {handshake.name = "muli2"} : index
        %17 = arith.addi %8, %16 {handshake.name = "addi3"} : index
        %18 = memref.load %arg0[%17] {handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.name = "load9"} : memref<400xi32>
        %19 = arith.addi %18, %15 {handshake.name = "addi1"} : i32
        %20 = arith.muli %arg4, %c20 {handshake.name = "muli3"} : index
        %21 = arith.addi %8, %20 {handshake.name = "addi4"} : index
        memref.store %19, %arg0[%21] {handshake.deps = #handshake<deps[["load8", 0, true], ["load9", 0, true], ["store1", 0, true]]>, handshake.name = "store1"} : memref<400xi32>
        %22 = arith.addi %arg5, %c1 {handshake.name = "addi5"} : index
        %23 = arith.cmpi ult, %22, %c20 {handshake.name = "cmpi0"} : index
        scf.condition(%23) {handshake.name = "condition0"} %22 : index
      } do {
      ^bb0(%arg5: index):
        scf.yield {handshake.name = "yield2"} %arg5 : index
      } attributes {handshake.name = "while0"}
      %5 = arith.addi %arg4, %c1 {handshake.name = "addi6"} : index
      %6 = arith.cmpi ult, %5, %c20 {handshake.name = "cmpi1"} : index
      scf.condition(%6) {handshake.name = "condition1"} %5 : index
    } do {
    ^bb0(%arg4: index):
      scf.yield {handshake.name = "yield3"} %arg4 : index
    } attributes {handshake.name = "while1"}
    return {handshake.name = "return0"}
  }
}

