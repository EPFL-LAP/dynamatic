module {
  func.func @jacobi_1d_imper(%arg0: memref<100xi32> {handshake.arg_name = "A"}, %arg1: memref<100xi32> {handshake.arg_name = "B"}) {
    %c-1 = arith.constant {handshake.name = "constant7"} -1 : index
    %c99 = arith.constant {handshake.name = "constant5"} 99 : index
    %c3_i32 = arith.constant {handshake.name = "constant0"} 3 : i32
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %c3 = arith.constant {handshake.name = "constant2"} 3 : index
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    %0 = scf.while (%arg2 = %c0) : (index) -> index {
      %1 = scf.while (%arg3 = %c1) : (index) -> index {
        %5 = arith.addi %arg3, %c-1 {handshake.name = "addi2"} : index
        %6 = memref.load %arg0[%5] {handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.name = "load0"} : memref<100xi32>
        %7 = memref.load %arg0[%arg3] {handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.name = "load1"} : memref<100xi32>
        %8 = arith.addi %6, %7 {handshake.name = "addi0"} : i32
        %9 = arith.addi %arg3, %c1 {handshake.name = "addi3"} : index
        %10 = memref.load %arg0[%9] {handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.name = "load2"} : memref<100xi32>
        %11 = arith.addi %8, %10 {handshake.name = "addi1"} : i32
        %12 = arith.muli %11, %c3_i32 {handshake.name = "muli0"} : i32
        memref.store %12, %arg1[%arg3] {handshake.deps = #handshake<deps[["store0", 1], ["load3", 1], ["load3", 2]]>, handshake.name = "store0"} : memref<100xi32>
        %13 = arith.addi %arg3, %c1 {handshake.name = "addi4"} : index
        %14 = arith.cmpi ult, %13, %c99 {handshake.name = "cmpi0"} : index
        scf.condition(%14) {handshake.name = "condition0"} %13 : index
      } do {
      ^bb0(%arg3: index):
        scf.yield {handshake.name = "yield3"} %arg3 : index
      } attributes {handshake.name = "while0"}
      %2 = scf.while (%arg3 = %c1) : (index) -> index {
        %5 = memref.load %arg1[%arg3] {handshake.deps = #handshake<deps[["store0", 1]]>, handshake.name = "load3"} : memref<100xi32>
        memref.store %5, %arg0[%arg3] {handshake.deps = #handshake<deps[["load0", 1], ["load1", 1], ["load2", 1], ["store1", 1]]>, handshake.name = "store1"} : memref<100xi32>
        %6 = arith.addi %arg3, %c1 {handshake.name = "addi5"} : index
        %7 = arith.cmpi ult, %6, %c99 {handshake.name = "cmpi1"} : index
        scf.condition(%7) {handshake.name = "condition1"} %6 : index
      } do {
      ^bb0(%arg3: index):
        scf.yield {handshake.name = "yield4"} %arg3 : index
      } attributes {handshake.name = "while1"}
      %3 = arith.addi %arg2, %c1 {handshake.name = "addi6"} : index
      %4 = arith.cmpi ult, %3, %c3 {handshake.name = "cmpi2"} : index
      scf.condition(%4) {handshake.name = "condition2"} %3 : index
    } do {
    ^bb0(%arg2: index):
      scf.yield {handshake.name = "yield5"} %arg2 : index
    } attributes {handshake.name = "while2"}
    return {handshake.name = "return0"}
  }
}

