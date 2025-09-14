module {
  handshake.func @sup_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["iterLiveIn"]} {
    %0:2 = fork [2] %arg2 {handshake.name = "fork_continue"} : <i1>
    %1 = spec_v2_repeating_init %0#0 {handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %2 = buffer %1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "buffer1"} : <i1>
    %3 = init %2 {handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %4 = mux %3 [%arg0, %arg1] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %5 = spec_v2_repeating_init %0#1 {handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    %6 = buffer %5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "buffer2"} : <i1>
    %7 = passer %4[%6] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %7 : <i1>
  }
}
