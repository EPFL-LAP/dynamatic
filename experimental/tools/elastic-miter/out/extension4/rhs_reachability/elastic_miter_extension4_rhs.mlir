module {
  handshake.func @extension4_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg", "ctrl"], resNames = ["res"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "arg_fork"} : <i1>
    %1:2 = fork [2] %arg1 {handshake.name = "ctrl_fork"} : <i1>
    %2 = andi %0#0, %1#0 {handshake.name = "andi"} : <i1>
    %3 = buffer %2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "and_buff"} : <i1>
    %4 = not %1#1 {handshake.name = "ctrl_not"} : <i1>
    %5 = ori %0#1, %4 {handshake.name = "ori"} : <i1>
    %6 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %7 = init %6 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %8:3 = fork [3] %7 {handshake.name = "ri_fork"} : <i1>
    %9 = source {handshake.name = "source1"} : <>
    %10 = constant %9 {handshake.name = "constant1", value = true} : <>, <i1>
    %11 = mux %8#0 [%10, %5] {handshake.name = "ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %12 = source {handshake.name = "source2"} : <>
    %13 = constant %12 {handshake.name = "constant2", value = true} : <>, <i1>
    %14 = mux %8#1 [%13, %3] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %15 = passer %8#2[%14] {handshake.name = "passer"} : <i1>, <i1>
    %16 = buffer %15, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "p_buff"} : <i1>
    end {handshake.name = "end0"} %16 : <i1>
  }
}
