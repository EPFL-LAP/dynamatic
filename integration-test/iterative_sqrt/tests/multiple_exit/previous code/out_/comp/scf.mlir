module {
  func.func @multiple_exit(%arg0: memref<10xi32> {handshake.arg_name = "arr"}, %arg1: i32 {handshake.arg_name = "size"}) {
    %true = arith.constant {handshake.name = "constant0"} true
    %c1_i32 = arith.constant {handshake.name = "constant1"} 1 : i32
    %false = arith.constant {handshake.name = "constant2"} false
    %c-1_i32 = arith.constant {handshake.name = "constant3"} -1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant4"} 0 : i32
    %0 = scf.while (%arg2 = %c0_i32, %arg3 = %true) : (i32, i1) -> i32 {
      %1 = arith.cmpi slt, %arg2, %arg1 {handshake.name = "cmpi0"} : i32
      %2 = arith.andi %1, %arg3 {handshake.name = "andi0"} : i1
      scf.condition(%2) {handshake.name = "condition0"} %arg2 : i32
    } do {
    ^bb0(%arg2: i32):
      %1 = arith.index_cast %arg2 {handshake.name = "index_cast0"} : i32 to index
      %2 = memref.load %arg0[%1] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load0"} : memref<10xi32>
      %3 = arith.cmpi ne, %2, %c-1_i32 {handshake.name = "cmpi1"} : i32
      %4:2 = scf.if %3 -> (i1, i32) {
        %5 = memref.load %arg0[%1] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load1"} : memref<10xi32>
        %6 = arith.cmpi ne, %5, %c0_i32 {handshake.name = "cmpi2"} : i32
        %7 = scf.if %6 -> (i32) {
          %8 = memref.load %arg0[%1] {handshake.deps = #handshake<deps[<"store0" (0)>]>, handshake.name = "load2"} : memref<10xi32>
          %9 = arith.addi %8, %c1_i32 {handshake.name = "addi0"} : i32
          memref.store %9, %arg0[%1] {handshake.deps = #handshake<deps[<"load0" (0)>, <"load1" (0)>, <"load2" (0)>, <"store0" (0)>]>, handshake.name = "store0"} : memref<10xi32>
          %10 = arith.addi %arg2, %c1_i32 {handshake.name = "addi1"} : i32
          scf.yield {handshake.name = "yield0"} %10 : i32
        } else {
          scf.yield {handshake.name = "yield1"} %arg2 : i32
        } {handshake.name = "if0"}
        scf.yield {handshake.name = "yield2"} %6, %7 : i1, i32
      } else {
        scf.yield {handshake.name = "yield3"} %false, %arg2 : i1, i32
      } {handshake.name = "if1"}
      scf.yield {handshake.name = "yield4"} %4#1, %4#0 : i32, i1
    } attributes {handshake.name = "while0"}
    return {handshake.name = "return0"}
  }
}

