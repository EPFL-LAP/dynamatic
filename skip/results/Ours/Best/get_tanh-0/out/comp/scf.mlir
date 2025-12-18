module {
  func.func @get_tanh(%arg0: memref<1000xf32> {handshake.arg_name = "A"}, %arg1: memref<1000xi32> {handshake.arg_name = "addr"}) {
    %cst = arith.constant {handshake.name = "constant0"} 3.70476198 : f32
    %cst_0 = arith.constant {handshake.name = "constant1"} 19.5238094 : f32
    %cst_1 = arith.constant {handshake.name = "constant2"} 1.000000e+00 : f32
    %c0 = arith.constant {handshake.name = "constant3"} 0 : index
    %c1000 = arith.constant {handshake.name = "constant4"} 1000 : index
    %c1 = arith.constant {handshake.name = "constant5"} 1 : index
    %0 = scf.while (%arg2 = %c0) : (index) -> index {
      %1 = memref.load %arg1[%arg2] {handshake.name = "load2"} : memref<1000xi32>
      %2 = arith.index_cast %1 {handshake.name = "index_cast0"} : i32 to index
      %3 = memref.load %arg0[%2] {handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.name = "load1"} : memref<1000xf32>
      %4 = arith.cmpf oge, %3, %cst_1 {handshake.name = "cmpf0"} : f32
      %5 = scf.if %4 -> (f32) {
        scf.yield {handshake.name = "yield0"} %cst_1 : f32
      } else {
        %8 = arith.mulf %3, %3 {handshake.name = "mulf0"} : f32
        %9 = arith.addf %8, %cst_0 {handshake.name = "addf0"} : f32
        %10 = arith.mulf %9, %3 {handshake.name = "mulf1"} : f32
        %11 = arith.mulf %10, %3 {handshake.name = "mulf2"} : f32
        %12 = arith.addf %11, %cst {handshake.name = "addf1"} : f32
        %13 = arith.mulf %12, %3 {handshake.name = "mulf3"} : f32
        scf.yield {handshake.name = "yield1"} %13 : f32
      } {handshake.name = "if0"}
      memref.store %5, %arg0[%2] {handshake.deps = #handshake<deps[["load1", 0, true], ["store0", 0, true]]>, handshake.name = "store0"} : memref<1000xf32>
      %6 = arith.addi %arg2, %c1 {handshake.name = "addi0"} : index
      %7 = arith.cmpi ult, %6, %c1000 {handshake.name = "cmpi0"} : index
      scf.condition(%7) {handshake.name = "condition0"} %6 : index
    } do {
    ^bb0(%arg2: index):
      scf.yield {handshake.name = "yield3"} %arg2 : index
    } attributes {handshake.name = "while0"}
    return {handshake.name = "return0"}
  }
}

