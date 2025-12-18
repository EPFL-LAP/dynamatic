module {
  func.func @gaussian(%arg0: memref<20xi32> {handshake.arg_name = "c"}, %arg1: memref<400xi32> {handshake.arg_name = "a"}) -> i32 {
    %c20 = arith.constant {handshake.name = "constant9"} 20 : index
    %c19 = arith.constant {handshake.name = "constant3"} 19 : index
    %c1 = arith.constant {handshake.name = "constant2"} 1 : index
    %c1_i32 = arith.constant {handshake.name = "constant0"} 1 : i32
    %c0_i32 = arith.constant {handshake.name = "constant1"} 0 : i32
    %0 = llvm.mlir.undef {handshake.name = "mlir.undef0"} : i32
    %1:2 = scf.while (%arg2 = %c1, %arg3 = %c0_i32) : (index, i32) -> (index, i32) {
      %2 = arith.addi %arg2, %c1 {handshake.name = "addi2"} : index
      %3 = scf.for %arg4 = %2 to %c19 step %c1 iter_args(%arg5 = %arg3) -> (i32) {
        %6:7 = scf.while (%arg6 = %c1, %arg7 = %c1_i32, %arg8 = %arg5, %arg9 = %arg5, %arg10 = %0, %arg11 = %c1_i32, %arg12 = %arg5) : (index, i32, i32, i32, i32, i32, i32) -> (index, i32, i32, i32, i32, i32, i32) {
          %7 = arith.index_cast %arg7 {handshake.name = "index_cast0"} : i32 to index
          %8 = arith.muli %arg4, %c20 {handshake.name = "muli1"} : index
          %9 = arith.addi %7, %8 {handshake.name = "addi3"} : index
          %10 = memref.load %arg1[%9] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load0"} : memref<400xi32>
          %11 = memref.load %arg0[%arg2] {handshake.name = "load1"} : memref<20xi32>
          %12 = arith.muli %arg2, %c20 {handshake.name = "muli2"} : index
          %13 = arith.addi %7, %12 {handshake.name = "addi4"} : index
          %14 = memref.load %arg1[%13] {handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load2"} : memref<400xi32>
          %15 = arith.muli %11, %14 {handshake.name = "muli0"} : i32
          %16 = arith.subi %10, %15 {handshake.name = "subi0"} : i32
          %17 = arith.muli %arg4, %c20 {handshake.name = "muli3"} : index
          %18 = arith.addi %7, %17 {handshake.name = "addi5"} : index
          memref.store %16, %arg1[%18] {handshake.deps = #handshake<deps[["load0", 0], ["load2", 0], ["store0", 0]]>, handshake.name = "store0"} : memref<400xi32>
          %19 = arith.addi %arg8, %arg7 {handshake.name = "addi0"} : i32
          %20 = arith.addi %arg7, %c1_i32 {handshake.name = "addi1"} : i32
          %21 = arith.addi %arg6, %c1 {handshake.name = "addi6"} : index
          %22 = arith.cmpi ult, %21, %c20 {handshake.name = "cmpi0"} : index
          scf.condition(%22) {handshake.name = "condition0"} %21, %20, %19, %19, %0, %20, %19 : index, i32, i32, i32, i32, i32, i32
        } do {
        ^bb0(%arg6: index, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32):
          scf.yield {handshake.name = "yield6"} %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12 : index, i32, i32, i32, i32, i32, i32
        } attributes {handshake.name = "while0"}
        scf.yield {handshake.name = "yield4"} %6#3 : i32
      } {handshake.name = "for4"}
      %4 = arith.addi %arg2, %c1 {handshake.name = "addi7"} : index
      %5 = arith.cmpi ult, %4, %c19 {handshake.name = "cmpi1"} : index
      scf.condition(%5) {handshake.name = "condition1"} %4, %3 : index, i32
    } do {
    ^bb0(%arg2: index, %arg3: i32):
      scf.yield {handshake.name = "yield7"} %arg2, %arg3 : index, i32
    } attributes {handshake.name = "while1"}
    return {handshake.name = "return0"} %1#1 : i32
  }
}

