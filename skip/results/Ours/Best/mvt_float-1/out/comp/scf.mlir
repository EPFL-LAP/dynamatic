module {
  func.func @mvt_float(%arg0: memref<900xf32> {handshake.arg_name = "A"}, %arg1: memref<30xf32> {handshake.arg_name = "x1"}, %arg2: memref<30xf32> {handshake.arg_name = "x2"}, %arg3: memref<30xf32> {handshake.arg_name = "y1"}, %arg4: memref<30xf32> {handshake.arg_name = "y2"}) {
    %c0 = arith.constant {handshake.name = "constant0"} 0 : index
    %c30 = arith.constant {handshake.name = "constant1"} 30 : index
    %c1 = arith.constant {handshake.name = "constant2"} 1 : index
    %0 = scf.while (%arg5 = %c0) : (index) -> index {
      %2 = memref.load %arg1[%arg5] {handshake.deps = #handshake<deps[["store2", 2, true]]>, handshake.name = "load6"} : memref<30xf32>
      %3:2 = scf.while (%arg6 = %c0, %arg7 = %2) : (index, f32) -> (index, f32) {
        %6 = arith.muli %arg5, %c30 {handshake.name = "muli0"} : index
        %7 = arith.addi %arg6, %6 {handshake.name = "addi0"} : index
        %8 = memref.load %arg0[%7] {handshake.name = "load12"} : memref<900xf32>
        %9 = memref.load %arg3[%arg6] {handshake.name = "load8"} : memref<30xf32>
        %10 = arith.mulf %8, %9 {handshake.name = "mulf0"} : f32
        %11 = arith.addf %arg7, %10 {handshake.name = "addf0"} : f32
        %12 = arith.addi %arg6, %c1 {handshake.name = "addi2"} : index
        %13 = arith.cmpi ult, %12, %c30 {handshake.name = "cmpi0"} : index
        scf.condition(%13) {handshake.name = "condition0"} %12, %11 : index, f32
      } do {
      ^bb0(%arg6: index, %arg7: f32):
        scf.yield {handshake.name = "yield6"} %arg6, %arg7 : index, f32
      } attributes {handshake.name = "while0"}
      memref.store %3#1, %arg1[%arg5] {handshake.name = "store2"} : memref<30xf32>
      %4 = arith.addi %arg5, %c1 {handshake.name = "addi3"} : index
      %5 = arith.cmpi ult, %4, %c30 {handshake.name = "cmpi1"} : index
      scf.condition(%5) {handshake.name = "condition1"} %4 : index
    } do {
    ^bb0(%arg5: index):
      scf.yield {handshake.name = "yield7"} %arg5 : index
    } attributes {handshake.name = "while1"}
    %1 = scf.while (%arg5 = %c0) : (index) -> index {
      %2 = memref.load %arg2[%arg5] {handshake.deps = #handshake<deps[["store3", 2, true]]>, handshake.name = "load9"} : memref<30xf32>
      %3:2 = scf.while (%arg6 = %c0, %arg7 = %2) : (index, f32) -> (index, f32) {
        %6 = arith.muli %arg6, %c30 {handshake.name = "muli1"} : index
        %7 = arith.addi %arg5, %6 {handshake.name = "addi1"} : index
        %8 = memref.load %arg0[%7] {handshake.name = "load13"} : memref<900xf32>
        %9 = memref.load %arg4[%arg6] {handshake.name = "load11"} : memref<30xf32>
        %10 = arith.mulf %8, %9 {handshake.name = "mulf1"} : f32
        %11 = arith.addf %arg7, %10 {handshake.name = "addf1"} : f32
        %12 = arith.addi %arg6, %c1 {handshake.name = "addi4"} : index
        %13 = arith.cmpi ult, %12, %c30 {handshake.name = "cmpi2"} : index
        scf.condition(%13) {handshake.name = "condition2"} %12, %11 : index, f32
      } do {
      ^bb0(%arg6: index, %arg7: f32):
        scf.yield {handshake.name = "yield8"} %arg6, %arg7 : index, f32
      } attributes {handshake.name = "while2"}
      memref.store %3#1, %arg2[%arg5] {handshake.name = "store3"} : memref<30xf32>
      %4 = arith.addi %arg5, %c1 {handshake.name = "addi5"} : index
      %5 = arith.cmpi ult, %4, %c30 {handshake.name = "cmpi3"} : index
      scf.condition(%5) {handshake.name = "condition3"} %4 : index
    } do {
    ^bb0(%arg5: index):
      scf.yield {handshake.name = "yield9"} %arg5 : index
    } attributes {handshake.name = "while3"}
    return {handshake.name = "return0"}
  }
}

