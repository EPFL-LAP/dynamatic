module {
  handshake.func @extension3_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg"], resNames = ["res"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "arg_fork"} : <i1>
    %1 = buffer %0#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "arg_buff"} : <i1>
    %2 = buffer %7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %3 = init %2 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %4:3 = fork [3] %3 {handshake.name = "ri_fork"} : <i1>
    %5 = source {handshake.name = "source1"} : <>
    %6 = constant %5 {handshake.name = "constant1", value = true} : <>, <i1>
    %7 = mux %4#0 [%6, %0#0] {handshake.name = "ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %8 = source {handshake.name = "source2"} : <>
    %9 = constant %8 {handshake.name = "constant2", value = true} : <>, <i1>
    %10 = mux %4#1 [%9, %1] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %11 = passer %4#2[%10] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %11 : <i1>
  }
}
