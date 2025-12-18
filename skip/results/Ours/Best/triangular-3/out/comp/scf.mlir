module {
  func.func @triangular(%arg0: memref<10xi32> {handshake.arg_name = "x"}, %arg1: i32 {handshake.arg_name = "n"}, %arg2: memref<100xi32> {handshake.arg_name = "a"}) {
    %c10 = arith.constant {handshake.name = "constant19"} 10 : index
    %c-2 = arith.constant {handshake.name = "constant8"} -2 : index
    %c-1 = arith.constant {handshake.name = "constant3"} -1 : index
    %c1 = arith.constant {handshake.name = "constant1"} 1 : index
    %c0 = arith.constant {handshake.name = "constant0"} 0 : index
    %0 = arith.index_cast %arg1 {handshake.name = "index_cast0"} : i32 to index
    scf.for %arg3 = %c0 to %0 step %c1 {
      %1 = arith.muli %arg3, %c-1 {handshake.name = "muli1"} : index
      %2 = arith.addi %1, %0 {handshake.name = "addi0"} : index
      %3 = arith.addi %2, %c-1 {handshake.name = "addi1"} : index
      scf.for %arg4 = %c0 to %3 step %c1 {
        %4 = arith.muli %arg3, %c-1 {handshake.name = "muli2"} : index
        %5 = arith.muli %arg4, %c-1 {handshake.name = "muli3"} : index
        %6 = arith.addi %4, %5 {handshake.name = "addi2"} : index
        %7 = arith.addi %6, %0 {handshake.name = "addi3"} : index
        %8 = arith.addi %7, %c-2 {handshake.name = "addi4"} : index
        %9 = arith.muli %arg3, %c-1 {handshake.name = "muli4"} : index
        %10 = arith.addi %9, %0 {handshake.name = "addi5"} : index
        %11 = arith.addi %10, %c-1 {handshake.name = "addi6"} : index
        %12 = arith.muli %8, %c10 {handshake.name = "muli10"} : index
        %13 = arith.addi %11, %12 {handshake.name = "addi15"} : index
        %14 = memref.load %arg2[%13] {handshake.name = "load6"} : memref<100xi32>
        %15 = arith.muli %arg3, %c-1 {handshake.name = "muli5"} : index
        %16 = arith.addi %15, %0 {handshake.name = "addi7"} : index
        %17 = arith.addi %16, %c-1 {handshake.name = "addi8"} : index
        %18 = memref.load %arg0[%17] {handshake.name = "load4"} : memref<10xi32>
        %19 = arith.muli %14, %18 {handshake.name = "muli0"} : i32
        %20 = arith.muli %arg3, %c-1 {handshake.name = "muli6"} : index
        %21 = arith.muli %arg4, %c-1 {handshake.name = "muli7"} : index
        %22 = arith.addi %20, %21 {handshake.name = "addi9"} : index
        %23 = arith.addi %22, %0 {handshake.name = "addi10"} : index
        %24 = arith.addi %23, %c-2 {handshake.name = "addi11"} : index
        %25 = arith.muli %24, %c10 {handshake.name = "muli11"} : index
        %26 = arith.addi %0, %25 {handshake.name = "addi16"} : index
        %27 = memref.load %arg2[%26] {handshake.deps = #handshake<deps[["store2", 1, true], ["store2", 3, true]]>, handshake.name = "load7"} : memref<100xi32>
        %28 = arith.subi %27, %19 {handshake.name = "subi0"} : i32
        %29 = arith.muli %arg3, %c-1 {handshake.name = "muli8"} : index
        %30 = arith.muli %arg4, %c-1 {handshake.name = "muli9"} : index
        %31 = arith.addi %29, %30 {handshake.name = "addi12"} : index
        %32 = arith.addi %31, %0 {handshake.name = "addi13"} : index
        %33 = arith.addi %32, %c-2 {handshake.name = "addi14"} : index
        %34 = arith.muli %33, %c10 {handshake.name = "muli12"} : index
        %35 = arith.addi %0, %34 {handshake.name = "addi17"} : index
        memref.store %28, %arg2[%35] {handshake.deps = #handshake<deps[["load7", 1, true], ["store2", 1, true]]>, handshake.name = "store2"} : memref<100xi32>
      } {handshake.name = "for2"}
    } {handshake.name = "for3"}
    return {handshake.name = "return0"}
  }
}

