module {
  func.func @atax(%arg0: memref<400xf32> {handshake.arg_name = "A"}, %arg1: memref<20xf32> {handshake.arg_name = "x"}, %arg2: memref<20xf32> {handshake.arg_name = "y"}, %arg3: memref<20xf32> {handshake.arg_name = "tmp"}) {
    %c0 = arith.constant {handshake.name = "constant0"} 0 : index
    %c20 = arith.constant {handshake.name = "constant1"} 20 : index
    %c1 = arith.constant {handshake.name = "constant2"} 1 : index
    %0 = scf.while (%arg4 = %c0) : (index) -> index {
      %1 = memref.load %arg3[%arg4] {handshake.deps = #handshake<deps[["store3", 2, true]]>, handshake.name = "load5"} : memref<20xf32>
      %2:2 = scf.while (%arg5 = %c0, %arg6 = %1) : (index, f32) -> (index, f32) {
        %6 = arith.muli %arg4, %c20 {handshake.name = "muli0"} : index
        %7 = arith.addi %arg5, %6 {handshake.name = "addi0"} : index
        %8 = memref.load %arg0[%7] {handshake.name = "load10"} : memref<400xf32>
        %9 = memref.load %arg1[%arg5] {handshake.name = "load7"} : memref<20xf32>
        %10 = arith.mulf %8, %9 {handshake.name = "mulf0"} : f32
        %11 = arith.addf %arg6, %10 {handshake.name = "addf0"} : f32
        %12 = arith.addi %arg5, %c1 {handshake.name = "addi2"} : index
        %13 = arith.cmpi ult, %12, %c20 {handshake.name = "cmpi0"} : index
        scf.condition(%13) {handshake.name = "condition0"} %12, %11 : index, f32
      } do {
      ^bb0(%arg5: index, %arg6: f32):
        scf.yield {handshake.name = "yield4"} %arg5, %arg6 : index, f32
      } attributes {handshake.name = "while0"}
      %3 = scf.while (%arg5 = %c0) : (index) -> index {
        %6 = memref.load %arg2[%arg5] {handshake.deps = #handshake<deps[["store2", 1, true], ["store2", 3, true]]>, handshake.name = "load8"} : memref<20xf32>
        %7 = arith.muli %arg4, %c20 {handshake.name = "muli1"} : index
        %8 = arith.addi %arg5, %7 {handshake.name = "addi1"} : index
        %9 = memref.load %arg0[%8] {handshake.name = "load11"} : memref<400xf32>
        %10 = arith.mulf %9, %2#1 {handshake.name = "mulf1"} : f32
        %11 = arith.addf %6, %10 {handshake.name = "addf1"} : f32
        memref.store %11, %arg2[%arg5] {handshake.deps = #handshake<deps[["load8", 1, true], ["store2", 1, true]]>, handshake.name = "store2"} : memref<20xf32>
        %12 = arith.addi %arg5, %c1 {handshake.name = "addi3"} : index
        %13 = arith.cmpi ult, %12, %c20 {handshake.name = "cmpi1"} : index
        scf.condition(%13) {handshake.name = "condition1"} %12 : index
      } do {
      ^bb0(%arg5: index):
        scf.yield {handshake.name = "yield5"} %arg5 : index
      } attributes {handshake.name = "while1"}
      memref.store %2#1, %arg3[%arg4] {handshake.name = "store3"} : memref<20xf32>
      %4 = arith.addi %arg4, %c1 {handshake.name = "addi4"} : index
      %5 = arith.cmpi ult, %4, %c20 {handshake.name = "cmpi2"} : index
      scf.condition(%5) {handshake.name = "condition2"} %4 : index
    } do {
    ^bb0(%arg4: index):
      scf.yield {handshake.name = "yield6"} %arg4 : index
    } attributes {handshake.name = "while2"}
    return {handshake.name = "return0"}
  }
}

