module {
  handshake.func @sup_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["iterLiveIn"]} {
    %0 = source {handshake.name = "source"} : <>
    %1 = constant %0 {handshake.name = "constant", value = true} : <>, <i1>
    %2 = mux %6 [%1, %arg2] {handshake.name = "ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %3:3 = fork [3] %2 {handshake.name = "fork_continue"} : <i1>
    %4 = init %3#0 {handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %5 = buffer %3#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %6 = init %5 {handshake.name = "newInit2", initToken = 0 : ui1} : <i1>
    %7 = mux %4 [%arg0, %arg1] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %8 = passer %7[%3#2] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %8 : <i1>
  }
}
