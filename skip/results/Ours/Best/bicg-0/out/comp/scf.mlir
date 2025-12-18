module {
  func.func @bicg(%arg0: memref<900xi32> {handshake.arg_name = "a"}, %arg1: memref<30xi32> {handshake.arg_name = "s"}, %arg2: memref<30xi32> {handshake.arg_name = "q"}, %arg3: memref<30xi32> {handshake.arg_name = "p"}, %arg4: memref<30xi32> {handshake.arg_name = "r"}) -> i32 {
    %c0_i32 = arith.constant {handshake.name = "constant0"} 0 : i32
    %c0 = arith.constant {handshake.name = "constant1"} 0 : index
    %c30 = arith.constant {handshake.name = "constant2"} 30 : index
    %c1 = arith.constant {handshake.name = "constant3"} 1 : index
    %0:2 = scf.while (%arg5 = %c0, %arg6 = %c0_i32) : (index, i32) -> (index, i32) {
      %1 = memref.load %arg2[%arg5] {handshake.deps = #handshake<deps[["store3", 2, true]]>, handshake.name = "load5"} : memref<30xi32>
      %2:2 = scf.while (%arg7 = %c0, %arg8 = %1) : (index, i32) -> (index, i32) {
        %5 = arith.muli %arg5, %c30 {handshake.name = "muli2"} : index
        %6 = arith.addi %arg7, %5 {handshake.name = "addi2"} : index
        %7 = memref.load %arg0[%6] {handshake.name = "load10"} : memref<900xi32>
        %8 = memref.load %arg1[%arg7] {handshake.deps = #handshake<deps[["store2", 1, true], ["store2", 3, true]]>, handshake.name = "load7"} : memref<30xi32>
        %9 = memref.load %arg4[%arg5] {handshake.name = "load8"} : memref<30xi32>
        %10 = arith.muli %9, %7 {handshake.name = "muli0"} : i32
        %11 = arith.addi %8, %10 {handshake.name = "addi0"} : i32
        memref.store %11, %arg1[%arg7] {handshake.deps = #handshake<deps[["load7", 1, true], ["store2", 1, true]]>, handshake.name = "store2"} : memref<30xi32>
        %12 = memref.load %arg3[%arg7] {handshake.name = "load9"} : memref<30xi32>
        %13 = arith.muli %7, %12 {handshake.name = "muli1"} : i32
        %14 = arith.addi %arg8, %13 {handshake.name = "addi1"} : i32
        %15 = arith.addi %arg7, %c1 {handshake.name = "addi3"} : index
        %16 = arith.cmpi ult, %15, %c30 {handshake.name = "cmpi0"} : index
        scf.condition(%16) {handshake.name = "condition0"} %15, %14 : index, i32
      } do {
      ^bb0(%arg7: index, %arg8: i32):
        scf.yield {handshake.name = "yield4"} %arg7, %arg8 : index, i32
      } attributes {handshake.name = "while0"}
      memref.store %2#1, %arg2[%arg5] {handshake.name = "store3"} : memref<30xi32>
      %3 = arith.addi %arg5, %c1 {handshake.name = "addi4"} : index
      %4 = arith.cmpi ult, %3, %c30 {handshake.name = "cmpi1"} : index
      scf.condition(%4) {handshake.name = "condition1"} %3, %2#1 : index, i32
    } do {
    ^bb0(%arg5: index, %arg6: i32):
      scf.yield {handshake.name = "yield5"} %arg5, %arg6 : index, i32
    } attributes {handshake.name = "while1"}
    return {handshake.name = "return0"} %0#1 : i32
  }
}

