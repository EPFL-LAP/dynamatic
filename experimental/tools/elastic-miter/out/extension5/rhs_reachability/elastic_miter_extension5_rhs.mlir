module {
  handshake.func @extension5_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg", "ctrl"], resNames = ["res"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "arg_fork"} : <i1>
    %1:2 = fork [2] %arg1 {handshake.name = "ctrl_fork"} : <i1>
    %2 = andi %0#0, %1#0 {handshake.name = "andi"} : <i1>
    %3 = not %1#1 {handshake.name = "ctrl_not"} : <i1>
    %4 = ori %0#1, %3 {handshake.name = "ori"} : <i1>
    %5 = buffer %4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "or_buff"} : <i1>
    %6:3 = fork [3] %11 {handshake.name = "ri_fork"} : <i1>
    %7 = buffer %6#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %8 = init %7 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %9 = source {handshake.name = "source1"} : <>
    %10 = constant %9 {handshake.name = "constant1", value = true} : <>, <i1>
    %11 = mux %8 [%10, %5] {handshake.name = "ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %12 = source {handshake.name = "source2"} : <>
    %13 = constant %12 {handshake.name = "constant2", value = true} : <>, <i1>
    %14 = mux %6#1 [%13, %2] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %15 = passer %6#2[%14] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %15 : <i1>
  }
}
