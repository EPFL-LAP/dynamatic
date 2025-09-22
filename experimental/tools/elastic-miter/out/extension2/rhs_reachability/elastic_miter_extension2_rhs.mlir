module {
  handshake.func @extension2_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg"], resNames = ["res"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "arg_fork"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "arg_buff"} : <i1>
    %2 = buffer %7#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %3 = init %2 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %4 = source {handshake.name = "source1"} : <>
    %5 = constant %4 {handshake.name = "constant1", value = true} : <>, <i1>
    %6 = mux %3 [%5, %1] {handshake.name = "ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %7:3 = fork [3] %6 {handshake.name = "ri_fork"} : <i1>
    %8 = source {handshake.name = "source2"} : <>
    %9 = constant %8 {handshake.name = "constant2", value = true} : <>, <i1>
    %10 = mux %7#1 [%9, %0#1] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %11 = passer %7#2[%10] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %11 : <i1>
  }
}
